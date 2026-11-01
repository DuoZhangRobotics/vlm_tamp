// Planning-only adapter around the pinned MC-VAMP implementation. No execution hooks.
// Upstream CRRTC, projection, FK, self collision and CAPT are used without edits.
#include <chrono>
#include <memory>
#include <limits>
#include <array>
#include <vector>
#include <cmath>
#include <cstring>
#include <Eigen/Geometry>
#include <vamp/robots/panda.hh>
#include <vamp/planning/crrtc.hh>
#include <vamp/planning/rrtc.hh>
#include <vamp/random/halton.hh>

namespace {
using Panda = vamp::robots::Panda;
using CAPT = vamp::collision::CAPT;
using Point = vamp::collision::Point;
using Clock = std::chrono::steady_clock;
constexpr size_t lanes = vamp::FloatVectorWidth;
// Upstream uses this threshold for both its convergence metric and joint-step
// stagnation check. Keep it unchanged and adapt only our task acceptance metric.
constexpr float constraint_metric_threshold = 1e-6F;
constexpr float angular_weighted_tolerance = 1e-6F;
using Block = Panda::ConfigurationBlock<lanes>;
// Sphere indices are the pinned upstream panda_spherized.urdf order:
// 0 fixed base; 1..36 arm; 37..54 palm; 55..58 fingers.
// Fingers have no environment veto, including stationary surfaces.
constexpr int first_arm_sphere=1, first_palm_sphere=37, first_finger_sphere=55;
struct Scene {
    std::shared_ptr<CAPT> stationary, moving, contact;
    Eigen::Vector3f origin, axis, anchor;
    Eigen::Matrix4f eef_to_tcp;
    Eigen::Vector3f pad_local;
    std::vector<std::array<float,5>> gripper; // upstream palm spheres in native EEF frame
    float point_radius=0.F;
    float position_tolerance=.003F;
    float orientation_tolerance=0.F;
    std::array<float,7> lower,upper;
    bool articulated=false, allow_contact=true, allow_palm_contact=false;
    float goal_angle=0;
    bool aligned_jaws=true;
    int grasp_constraint=0;
    Eigen::Matrix3f reference_rotation;
    Clock::time_point deadline;
    bool timed_out=false;
    size_t checks=0,self_rejects=0,static_rejects=0,moving_rejects=0,bounds_rejects=0;
    int last_part=-1,last_surface=-1;
};
thread_local Scene* active=nullptr;

std::shared_ptr<CAPT> tree(const float* xyz,int n,float radius) {
    if(!n) return {};
    std::vector<Point> points;points.reserve(n);
    for(int i=0;i<n;++i) points.push_back({xyz[3*i],xyz[3*i+1],xyz[3*i+2]});
    // Upstream collides_simd reads the root split even for a one-point tree.
    // Repeating that same point supplies a split without changing geometry.
    if(points.size()==1) points.push_back(points.front());
    return std::make_shared<CAPT>(points,Panda::min_radius,Panda::max_radius,radius);
}

bool gripper_scene_clear(Scene& s,const Eigen::Matrix3f& rotation,
                         const Eigen::Vector3f& translation) {
    const std::array<std::shared_ptr<CAPT>,3> trees={s.stationary,s.moving,s.contact};
    for(size_t i=0;i<s.gripper.size();++i) {
        const auto& g=s.gripper[i];
        const Eigen::Vector3f center=rotation*Eigen::Vector3f(g[0],g[1],g[2])+translation;
        for(int surface=0;surface<3;++surface) {
            if(!trees[surface] || (surface==2 && s.allow_palm_contact)) continue;
            if(trees[surface]->collides(Point{center.x(),center.y(),center.z()},g[3])) {
                s.last_part=first_palm_sphere+int(i);s.last_surface=surface;
                if(surface==0) ++s.static_rejects;else ++s.moving_rejects;
                return false;
            }
        }
    }
    s.last_part=s.last_surface=-1;return true;
}

struct ObservedPanda : Panda {
    template<size_t rake>
    static bool fkcc(const vamp::collision::Environment<vamp::FloatVector<rake>>&,
                     const ConfigurationBlock<rake>& q) noexcept {
        static_assert(rake==lanes,"CAPT uses the upstream native SIMD width");
        using V=vamp::FloatVector<rake>;
        Scene& s=*active;++s.checks;s.last_part=s.last_surface=-1;
        if(Clock::now()>s.deadline) {s.timed_out=true;return false;}
        for(size_t lane=0;lane<rake;++lane) for(size_t j=0;j<7;++j)
            if(!std::isfinite(q[{j,lane}]) || q[{j,lane}]<s.lower[j] || q[{j,lane}]>s.upper[j]) {
                ++s.bounds_rejects;return false;
            }
        // Upstream's sphere-based self-collision check, without a replacement.
        static const vamp::collision::Environment<V> empty;
        if(!Panda::fkcc<rake>(empty,q)) {++s.self_rejects;return false;}
        typename Panda::template Spheres<rake> spheres;
        Panda::sphere_fk<rake>(q,spheres);
        // Only the moving-cloud coordinate transform is application-specific.
        // Keep CAPT environment queries vectorized across the planner's states.
        alignas(32) std::array<std::array<float,rake>,12> transforms{};
        if(s.articulated) for(size_t lane=0;lane<rake;++lane) {
            std::array<float,7> scalar;
            for(size_t j=0;j<7;++j) scalar[j]=q[{j,lane}];
            const Eigen::Matrix4f tcp=Panda::eefk(scalar)[0].matrix()*s.eef_to_tcp;
            const Eigen::Vector3f point=tcp.block<3,1>(0,3)+tcp.block<3,3>(0,0)*s.pad_local;
            Eigen::Vector3f first=s.anchor-s.origin,now=point-s.origin;
            first-=s.axis*first.dot(s.axis);now-=s.axis*now.dot(s.axis);
            const float angle=std::atan2(s.axis.dot(first.cross(now)),first.dot(now));
            const Eigen::Matrix3f r=Eigen::AngleAxisf(-angle,s.axis).toRotationMatrix();
            const Eigen::Vector3f t=s.origin-r*s.origin;
            for(int j=0;j<3;++j) {
                for(int k=0;k<3;++k) transforms[4*j+k][lane]=r(j,k);
                transforms[4*j+3][lane]=t[j];
            }
        }
        std::array<V,12> transform;
        if(s.articulated) for(int i=0;i<12;++i) transform[i]=V(transforms[i]);
        for(int i=first_arm_sphere;i<first_finger_sphere;++i) {
            std::array<V,3> centers={V(spheres.x[i]),V(spheres.y[i]),V(spheres.z[i])};
            const V radius(spheres.r[i]);
            auto blocked=[&](int surface,const std::shared_ptr<CAPT>& cloud,const std::array<V,3>& points) {
                if(!cloud || (surface==2 && i>=first_palm_sphere && s.allow_palm_contact)) return false;
                if(!cloud->collides_simd(points,radius)) return false;
                s.last_part=i;s.last_surface=surface;
                if(surface==0) ++s.static_rejects;else ++s.moving_rejects;
                return true;
            };
            if(blocked(0,s.stationary,centers)) return false;
            if(s.articulated) {
                const auto original=centers;
                for(int j=0;j<3;++j) centers[j]=transform[4*j]*original[0]+transform[4*j+1]*original[1]
                    +transform[4*j+2]*original[2]+transform[4*j+3];
            }
            if(blocked(1,s.moving,centers) || blocked(2,s.contact,centers)) return false;
        }
        s.last_part=s.last_surface=-1;return true;
    }
    template<size_t rake>
    static bool fkcc_attach(const vamp::collision::Environment<vamp::FloatVector<rake>>& env,
                            const ConfigurationBlock<rake>& q) noexcept {return fkcc<rake>(env,q);}
};
// A seven-joint representation of the existing point-on-hinge-arc constraint.
// Door progress is derived from the measured grip point, rather than becoming
// an eighth sampled coordinate. Projection is called by upstream MC-VAMP.
class Task {
    Scene geometry;
public:
    explicit Task(const Scene& s) : geometry() {
        geometry.origin=s.origin;geometry.axis=s.axis;geometry.anchor=s.anchor;
        geometry.eef_to_tcp=s.eef_to_tcp;geometry.pad_local=s.pad_local;
        geometry.goal_angle=s.goal_angle;geometry.aligned_jaws=s.aligned_jaws;
        geometry.reference_rotation=s.reference_rotation;
        geometry.grasp_constraint=s.grasp_constraint;
        geometry.position_tolerance=s.position_tolerance;
        geometry.orientation_tolerance=s.orientation_tolerance;
    }
    Task(const Task& other): Task(other.geometry) {}
    using Residual = Eigen::Matrix<float,6,1>;
    Residual error(const std::array<float,7>& q) const {
        const auto& s=geometry;
        const Eigen::Matrix4f tcp=Panda::eefk(q)[0].matrix()*s.eef_to_tcp;
        const Eigen::Vector3f point=tcp.block<3,1>(0,3)+tcp.block<3,3>(0,0)*s.pad_local;
        Eigen::Vector3f first=s.anchor-s.origin,now=point-s.origin;
        const float height=now.dot(s.axis)-first.dot(s.axis);
        first-=s.axis*first.dot(s.axis);now-=s.axis*now.dot(s.axis);
        const float theta=std::atan2(s.axis.dot(first.cross(now)),first.dot(now));
        Residual result=Residual::Zero();
        result[0]=now.norm()-first.norm();result[1]=height;
        if(s.grasp_constraint) {
            const Eigen::Matrix3f expected=Eigen::AngleAxisf(theta,s.axis).toRotationMatrix()*s.reference_rotation;
            const Eigen::Matrix3f actual=tcp.block<3,3>(0,0);
            if(s.grasp_constraint==1) {
                Eigen::AngleAxisf error(expected.transpose()*actual);
                result.segment<3>(2)=.15F*error.angle()*error.axis();
            } else {
                result.segment<3>(2)=.15F*(actual.col(0)-expected.col(0));
            }
            result[5]=.15F*(theta-std::clamp(theta,-.03F,s.goal_angle+.03F));
            return result;
        }
        // Point mode constrains only the grip-point circle and requested arc extent.
        result[3]=.15F*(theta-std::clamp(theta,-.03F,s.goal_angle+.03F));
        return result;
    }
    float acceptanceMetric(const Residual& residual) const {
        // The radial/height pair is the Euclidean distance to the hinge circle.
        // Orientation and progress bounds have separate tolerances. This metric
        // must not be interpreted as a distance in metres. Keep error() and its
        // Jacobian in their original scale to preserve projection conditioning.
        if(!residual.allFinite()) return std::numeric_limits<float>::infinity();
        float angular_ratio=residual.tail<4>().norm()/angular_weighted_tolerance;
        if(geometry.grasp_constraint && geometry.orientation_tolerance>0.F) {
            const float tolerance=.15F*(geometry.grasp_constraint==1 ? geometry.orientation_tolerance
                : 2.F*std::sin(geometry.orientation_tolerance/2.F));
            angular_ratio=std::max(residual.segment<3>(2).norm()/tolerance,
                                  std::abs(residual[5])/angular_weighted_tolerance);
        }
        const float ratio=std::max(residual.head<2>().norm()/geometry.position_tolerance,angular_ratio);
        return constraint_metric_threshold*ratio;
    }
    vamp::FloatVector<lanes,1> distanceToConstraint(const Block& q) const {
        alignas(32) std::array<float,lanes> result{};
        for(size_t lane=0;lane<lanes;++lane) {
            std::array<float,7> scalar;
            for(size_t j=0;j<7;++j) scalar[j]=q[{j,lane}];
            result[lane]=acceptanceMetric(error(scalar));
        }
        return vamp::FloatVector<lanes,1>(result);
    }
    void projectStepInPlace(Block& q,vamp::planning::constraint::ProjMethod,float alpha) {
        alignas(32) std::array<std::array<float,lanes>,7> values{};
        for(size_t j=0;j<7;++j) for(size_t lane=0;lane<lanes;++lane) values[j][lane]=q[{j,lane}];
        for(size_t lane=0;lane<lanes;++lane) {
            std::array<float,7> scalar;
            for(size_t j=0;j<7;++j) scalar[j]=q[{j,lane}];
            const Residual residual=error(scalar);
            if(acceptanceMetric(residual)<=constraint_metric_threshold) continue;
            Eigen::Matrix<float,6,7> jacobian;
            // Float FK needs a milliradian difference. This is planning-only
            // projection; expose the adapter cost separately in timing reports.
            for(size_t j=0;j<7;++j) {
                auto a=scalar,b=scalar;a[j]+=.001F;b[j]-=.001F;
                jacobian.col(j)=(error(a)-error(b))/.002F;
            }
            Eigen::Matrix<float,6,6> gram=jacobian*jacobian.transpose();
            gram.diagonal().array()+=1e-8F;
            Eigen::Matrix<float,7,1> step=jacobian.transpose()*gram.ldlt().solve(residual);
            const float largest=step.cwiseAbs().maxCoeff();
            if(largest>.20F) step*=.20F/largest;
            for(size_t j=0;j<7;++j) values[j][lane]-=alpha*step[j];
        }
        for(size_t j=0;j<7;++j) q[j]=vamp::FloatVector<lanes>(values[j]);
    }
    void print_robot_tsr_error(const Block&) const {}
};
using Constraint = vamp::planning::constraint::ComposableConstraints<ObservedPanda, lanes, Task>;
Block block(const float* q) {
    Block b;
    for (size_t j=0;j<7;++j) b[j] = vamp::FloatVector<lanes>::fill(q[j]);
    return b;
}
std::unique_ptr<Constraint> constraint(const float*) {
    return std::make_unique<Constraint>(Task(*active));
}
void statistics(Scene& s, double* out) {
    out[0]=s.checks; out[1]=s.self_rejects; out[2]=s.static_rejects;
    out[3]=s.moving_rejects; out[4]=s.bounds_rejects; out[5]=s.timed_out;
    out[6]=s.last_part; out[7]=s.last_surface;
}
} // namespace

extern "C" {
void mcvamp_configure(Scene* s,const float* config,const float* gripper,int ng) {
    s->origin=Eigen::Map<const Eigen::Vector3f>(config);
    s->axis=Eigen::Map<const Eigen::Vector3f>(config+3);
    s->anchor=Eigen::Map<const Eigen::Vector3f>(config+6);
    s->eef_to_tcp=Eigen::Map<const Eigen::Matrix<float,4,4,Eigen::RowMajor>>(config+9);
    s->pad_local=Eigen::Map<const Eigen::Vector3f>(config+25);
    std::copy_n(config+28,7,s->lower.begin()); std::copy_n(config+35,7,s->upper.begin());
    s->articulated=config[42]>.5; s->allow_contact=config[43]>.5;
    std::array<float,7> reference;
    std::copy_n(config+44,7,reference.begin());
    s->reference_rotation=(Panda::eefk(reference)[0].matrix()*s->eef_to_tcp).block<3,3>(0,0);
    s->goal_angle=config[51];s->aligned_jaws=config[52]>.5;s->grasp_constraint=0;
    s->gripper.resize(ng);
    for(int i=0;i<ng;++i) std::copy_n(gripper+5*i,5,s->gripper[i].begin());
    s->deadline=Clock::now()+std::chrono::hours(1);
    s->checks=s->self_rejects=s->static_rejects=s->moving_rejects=s->bounds_rejects=0;
    s->timed_out=false;
}
void* mcvamp_create(const float* stationary,int ns,const float* moving,int nm,const float* contact,int nc,
                    const float* config,const float* gripper,int ng,float point_radius) {
    try {
        auto s=std::make_unique<Scene>();
        s->stationary=tree(stationary,ns,point_radius);
        s->moving=tree(moving,nm,point_radius);
        s->contact=tree(contact,nc,point_radius);
        s->point_radius=point_radius;
        mcvamp_configure(s.get(),config,gripper,ng);
        return s.release();
    } catch (...) { return nullptr; }
}
void* mcvamp_clone(void* original,const float* config,const float* gripper,int ng) {
    try {
        auto s=std::make_unique<Scene>(*static_cast<Scene*>(original));
        mcvamp_configure(s.get(),config,gripper,ng);
        return s.release();
    } catch (...) { return nullptr; }
}
void* mcvamp_create_reusing_static(void* original,const float* moving,int nm,const float* contact,int nc,
                                  const float* config,const float* gripper,int ng) {
    try {
        if(!original) return nullptr;
        const auto& previous=*static_cast<Scene*>(original);
        auto s=std::make_unique<Scene>();
        s->stationary=previous.stationary;
        s->point_radius=previous.point_radius;
        s->moving=tree(moving,nm,s->point_radius);
        s->contact=tree(contact,nc,s->point_radius);
        mcvamp_configure(s.get(),config,gripper,ng);
        return s.release();
    } catch (...) { return nullptr; }
}
void mcvamp_destroy(void* p) { delete static_cast<Scene*>(p); }
int mcvamp_set_grasp_constraint(void* p,int mode) {
    if(mode<0 || mode>2) return 0;
    static_cast<Scene*>(p)->grasp_constraint=mode;return 1;
}
void mcvamp_set_palm_contact(void* p,int enabled) {
    static_cast<Scene*>(p)->allow_palm_contact=bool(enabled);
}
int mcvamp_set_position_tolerance(void* p,float tolerance) {
    if(!std::isfinite(tolerance) || tolerance<=0.F) return 0;
    static_cast<Scene*>(p)->position_tolerance=tolerance;return 1;
}
int mcvamp_set_orientation_tolerance(void* p,float tolerance) {
    if(!std::isfinite(tolerance) || tolerance<0.F || tolerance>float(M_PI)) return 0;
    static_cast<Scene*>(p)->orientation_tolerance=tolerance;return 1;
}
// Return physical residual components independently of the scaled acceptance
// metric, so callers can report millimetres without conflating angular terms.
int mcvamp_constraint_error(void* p,const float* q,float* error,double* metric) {
    Task task(*static_cast<Scene*>(p));
    std::array<float,7> scalar;std::copy_n(q,7,scalar.begin());
    const auto residual=task.error(scalar);
    std::copy_n(residual.data(),6,error);
    *metric=task.acceptanceMetric(residual);
    return std::isfinite(*metric) && *metric<=constraint_metric_threshold;
}
int mcvamp_validate(void* p,const float* q,double* stats) {
    auto& s=*static_cast<Scene*>(p); active=&s; s.deadline=Clock::now()+std::chrono::seconds(60);
    const vamp::collision::Environment<vamp::FloatVector<lanes>> empty;
    const bool ok=ObservedPanda::fkcc<lanes>(empty,block(q));
    statistics(s,stats); return ok;
}
int mcvamp_validate_hand_pose(void* p,const float* eef_transform,double* stats) {
    auto& s=*static_cast<Scene*>(p);
    // This filter applies to a fixed observed scene, before an approach IK call.
    if(s.articulated) return -1;
    const Eigen::Map<const Eigen::Matrix<float,4,4,Eigen::RowMajor>> pose(eef_transform);
    if(!pose.allFinite()) return -1;
    ++s.checks;
    const bool ok=gripper_scene_clear(s,pose.block<3,3>(0,0),pose.block<3,1>(0,3));
    statistics(s,stats);return ok;
}
// Project and test candidates during planning; never called by the executor.
int mcvamp_project(void* p,const float* tsr,const float* q,float* result,double* residual) {
    auto& s=*static_cast<Scene*>(p); active=&s;
    auto c=constraint(tsr); auto b=block(q), projected=b;
    c->projectConfiguration(b,projected,vamp::planning::constraint::InnerLM,.5F,1.F,100,false);
    for(size_t j=0;j<7;++j) result[j]=projected[{j,0}];
    *residual=c->distanceToConstraint(projected)[{0,0}];
    return std::isfinite(*residual) && *residual<=constraint_metric_threshold;
}
// Return the upstream planner solution without a second validation/resampling pass.
// stats[8:]: solve seconds, post-validation seconds (zero), iterations, path count, returned count.
int mcvamp_plan(void* p,const float* start,const float* goal,const float* tsr,int constrained,
                 int max_iterations,double seconds,float* output,int capacity,double* stats) {
    try {
        auto& s=*static_cast<Scene*>(p); active=&s;
        s.checks=s.self_rejects=s.static_rejects=s.moving_rejects=s.bounds_rejects=0;
        s.timed_out=false; s.deadline=Clock::now()+std::chrono::milliseconds(int(seconds*1000));
        const vamp::collision::Environment<vamp::FloatVector<lanes>> empty;
        // NumPy/ctypes inputs need not be 32-byte aligned. Native vector
        // constructors and to_array use aligned AVX loads/stores.
        alignas(32) std::array<float,8> start_buffer{},goal_buffer{};
        std::copy_n(start,7,start_buffer.begin());std::copy_n(goal,7,goal_buffer.begin());
        Panda::Configuration a(start_buffer.data()),b(goal_buffer.data());
        auto rng=std::make_shared<vamp::rng::Halton<ObservedPanda>>();
        std::unique_ptr<Constraint> c;
        vamp::planning::PlanningResult<ObservedPanda> solved;
        const auto before=Clock::now();
        if(constrained) {
            c=constraint(tsr);
            vamp::planning::CRRTCSettings settings;
            settings.rrtc_settings.range=.15F; settings.rrtc_settings.max_iterations=max_iterations;
            settings.rrtc_settings.max_samples=100000;
            settings.constraint_settings.num_projection_iterations=100;
            settings.constraint_settings.std_dev_scaling_factor=.05F;
            solved=vamp::planning::CRRTC<ObservedPanda,lanes,32,Task>::solve(a,b,empty,settings,*c,rng);
        } else {
            vamp::planning::RRTCSettings settings;
            settings.range=.15F;settings.max_iterations=max_iterations;settings.max_samples=100000;
            // Use ordinary RRT-Connect exploration for near-contact approaches.
            // Dynamic-domain pruning can reject all later samples after a few
            // failed expansions, exhausting iterations without testing motion.
            settings.dynamic_domain=false;
            solved=vamp::planning::RRTC<ObservedPanda,lanes,32>::solve(a,b,empty,settings,rng);
        }
        const auto resample=Clock::now();
        stats[8]=std::chrono::duration<double>(resample-before).count();
        stats[10]=solved.iterations; stats[11]=solved.path.size(); stats[12]=0;
        if(solved.path.size()<2) { statistics(s,stats); return 0; }
        // Trust upstream's checked solution. The executor only time-parameterizes
        // these segments; this adapter adds no dense collision/projection pass.
        if(solved.path.size()>static_cast<size_t>(capacity)) {
            statistics(s,stats); return -6;
        }
        alignas(32) std::array<float,8> q{};
        int n=0;
        for(const auto& waypoint:solved.path) {
            waypoint.to_array(q.data());
            std::copy_n(q.data(),7,output+n*7);++n;
        }
        stats[9]=0.;stats[12]=n;
        statistics(s,stats);return n;
    } catch (...) { return -9; }
}
}
