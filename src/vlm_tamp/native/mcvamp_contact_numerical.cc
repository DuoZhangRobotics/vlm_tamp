// Planning-only adapter around the pinned MC-VAMP implementation. No execution hooks.
// Upstream CRRTC, projection, FK, self collision and CAPT are used without edits.
#include <chrono>
#include <memory>
#include <array>
#include <vector>
#include <cmath>
#include <cstring>
#include <unordered_map>
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
using Block = Panda::ConfigurationBlock<lanes>;
struct CellHash {
    size_t operator()(const std::array<int,3>& key) const noexcept {
        size_t h=0;
        for(int k:key) h^=std::hash<int>{}(k)+0x9e3779b9+(h<<6)+(h>>2);
        return h;
    }
};
struct PointGrid {
    static constexpr float spacing=.03F;
    std::unordered_map<std::array<int,3>,std::vector<Eigen::Vector3f>,CellHash> cells;
    PointGrid(const float* xyz,int n) {
        for(int i=0;i<n;++i) {
            Eigen::Vector3f p=Eigen::Map<const Eigen::Vector3f>(xyz+3*i);
            std::array<int,3> key;
            for(int j=0;j<3;++j) key[j]=int(std::floor(p[j]/spacing));
            cells[key].push_back(p);
        }
    }
};
struct ConvexGripper {
    std::vector<Eigen::Vector4f> planes;
    Eigen::Vector3f center;
    float radius=0;
    bool finger=false;
    int first=0,end=0;
};
struct Scene {
    std::shared_ptr<CAPT> stationary, moving, contact;
    std::array<std::shared_ptr<PointGrid>,3> point_grids;
    Eigen::Vector3f origin, axis, anchor;
    Eigen::Matrix4f eef_to_tcp;
    Eigen::Vector3f pad_local;
    std::vector<std::array<float, 5>> gripper; // native EEF-frame xyz, radius, finger flag
    std::vector<ConvexGripper> gripper_convex;
    std::vector<ConvexGripper> arm_convex;
    std::array<Eigen::Vector3f,7> joint_axes, joint_anchors;
    std::array<float,7> calibration_q;

    float point_radius=.0025F;
    float gripper_point_padding=0.F;
    float existing_palm_tolerance=0.F;
    std::array<float, 7> lower, upper;
    bool articulated = false;
    bool allow_contact = true;
    float goal_angle = 0;
    bool aligned_jaws = true;
    int grasp_constraint = 0; // 0 point, 1 rigid relative pose, 2 contact-line direction
    Eigen::Matrix3f reference_rotation;
    Clock::time_point deadline;
    bool timed_out = false;
    size_t checks = 0, self_rejects = 0, static_rejects = 0, moving_rejects = 0;
    size_t bounds_rejects = 0;
    int last_part = -1, last_surface = -1;
};
thread_local Scene* active = nullptr;

bool convex_collision(const Scene& s,const ConvexGripper& volume,int surface,
                      const Eigen::Matrix3f& rotation,const Eigen::Vector3f& translation) {
    const auto& grid=s.point_grids[surface];
    if(!grid) return false;
    const Eigen::Vector3f center=rotation*volume.center+translation;
    std::array<int,3> lo,hi;
    for(int j=0;j<3;++j) {
        lo[j]=int(std::floor((center[j]-volume.radius)/PointGrid::spacing));
        hi[j]=int(std::floor((center[j]+volume.radius)/PointGrid::spacing));
    }
    for(int x=lo[0];x<=hi[0];++x) for(int y=lo[1];y<=hi[1];++y) for(int z=lo[2];z<=hi[2];++z) {
        auto cell=grid->cells.find({x,y,z});
        if(cell==grid->cells.end()) continue;
        for(const auto& point:cell->second) {
            if((point-center).squaredNorm()>volume.radius*volume.radius) continue;
            const Eigen::Vector3f local=rotation.transpose()*(point-translation);
            bool inside=true;
            for(const auto& plane:volume.planes)
                if(plane.head<3>().dot(local)+plane[3]>((surface==2 && !volume.finger && volume.first==0) ? -s.existing_palm_tolerance : s.gripper_point_padding)) {inside=false;break;}
            if(inside) return true;
        }
    }
    return false;
}

std::shared_ptr<CAPT> tree(const float* xyz, int n, float radius) {
    if (!n) return {};
    std::vector<Point> points;
    points.reserve(n);
    for (int i = 0; i < n; ++i) points.push_back({xyz[3*i], xyz[3*i+1], xyz[3*i+2]});
    // Tight query-radius bounds matter enormously for CAPT's affordance size.
    // The native arm max is 8 cm; calibrated gripper cells are below 2 cm.
    return std::make_shared<CAPT>(points, .006F, .081F, radius);
}

struct ObservedPanda : Panda {
    template<size_t rake>
    static bool fkcc(const vamp::collision::Environment<vamp::FloatVector<rake>>&,
                     const ConfigurationBlock<rake>& q) noexcept {
        Scene& s = *active;
        ++s.checks;
        if (Clock::now() > s.deadline) { s.timed_out = true; return false; }
        for (size_t lane = 0; lane < rake; ++lane)
            for (size_t j = 0; j < 7; ++j)
                if (!std::isfinite(q[{j,lane}]) || q[{j,lane}] < s.lower[j] || q[{j,lane}] > s.upper[j]) {
                    ++s.bounds_rejects; return false;
                }
        static const vamp::collision::Environment<vamp::FloatVector<rake>> empty;
        if (!Panda::fkcc<rake>(empty, q)) { ++s.self_rejects; return false; }
        typename Panda::template Spheres<rake> spheres;
        Panda::sphere_fk<rake>(q, spheres);
        for (size_t lane = 0; lane < rake; ++lane) {
            std::array<float,7> scalar;
            for (size_t j = 0; j < 7; ++j) scalar[j] = q[{j,lane}];
            const auto eef = Panda::eefk(scalar)[0];
            Eigen::Matrix3f inverse_rotation = Eigen::Matrix3f::Identity();
            if (s.articulated) {
                Eigen::Matrix4f tcp = eef.matrix() * s.eef_to_tcp;
                Eigen::Vector3f point = tcp.block<3,1>(0,3) + tcp.block<3,3>(0,0)*s.pad_local;
                Eigen::Vector3f first = s.anchor-s.origin, now = point-s.origin;
                first -= s.axis*first.dot(s.axis); now -= s.axis*now.dot(s.axis);
                const float theta = std::atan2(s.axis.dot(first.cross(now)), first.dot(now));
                inverse_rotation = Eigen::AngleAxisf(-theta, s.axis).toRotationMatrix();
            }
            auto collision = [&](const Eigen::Vector3f& center, float radius, bool finger, int part) {
                s.last_part = part;
                Point p{center.x(), center.y(), center.z()};
                if (s.stationary && s.stationary->collides(p, radius)) {
                    ++s.static_rejects; s.last_surface = 0; return true;
                }
                Eigen::Vector3f local = s.origin + inverse_rotation*(center-s.origin);
                Point m{local.x(), local.y(), local.z()};
                if (s.moving && s.moving->collides(m, radius)) {
                    ++s.moving_rejects; s.last_surface = 1; return true;
                }
                if ((!finger || !s.allow_contact) && s.contact && s.contact->collides(m, radius)) {
                    ++s.moving_rejects; s.last_surface = 2; return true;
                }
                return false;
            };
            // The URDF's arm occupies spheres 0..36. Its hard-coded open hand
            // and fingers (37..58) are replaced by calibrated configured geometry.
            // The fixed base sphere is omitted, as in the existing scene checker.
            if(s.arm_convex.empty()) return false; // Calibrated arm geometry is mandatory.
            std::array<Eigen::Matrix4f,8> prefixes;
            prefixes[0]=Eigen::Matrix4f::Identity();
            for(int j=0;j<7;++j) {
                Eigen::Matrix4f step=Eigen::Matrix4f::Identity();
                const Eigen::Matrix3f r=Eigen::AngleAxisf(scalar[j]-s.calibration_q[j],s.joint_axes[j]).toRotationMatrix();
                step.block<3,3>(0,0)=r;
                step.block<3,1>(0,3)=s.joint_anchors[j]-r*s.joint_anchors[j];
                prefixes[j+1]=prefixes[j]*step;
            }
            for(const auto& volume:s.arm_convex) {
                for(int surface=0;surface<3;++surface) {
                    Eigen::Matrix3f r=prefixes[volume.first].block<3,3>(0,0);
                    Eigen::Vector3f t=prefixes[volume.first].block<3,1>(0,3);
                    if(surface) {r=inverse_rotation*r;t=s.origin+inverse_rotation*(t-s.origin);}
                    if(convex_collision(s,volume,surface,r,t)) {
                        s.last_part=2000+volume.first;s.last_surface=surface;
                        if(surface==0) ++s.static_rejects;else ++s.moving_rejects;
                        return false;
                    }
                }
            }
            // Cell balls are only a broad phase. Their overhang must not become
            // physical palm/finger geometry. Confirm hits against the calibrated
            // convex volume. Gripper padding is explicit and independently
            // configurable; contact-rich runs use the configured boundary.
            for(const auto& volume:s.gripper_convex) {
                for(int surface=0;surface<3;++surface) {
                    if(surface==2 && volume.finger && s.allow_contact) continue;
                    const auto& index=(surface==0?s.stationary:surface==1?s.moving:s.contact);
                    if(!index) continue;
                    Eigen::Matrix3f rotation=eef.linear();
                    Eigen::Vector3f translation=eef.translation();
                    if(surface) {
                        rotation=inverse_rotation*rotation;
                        translation=s.origin+inverse_rotation*(translation-s.origin);
                    }
                    bool broad=false;
                    for(int i=volume.first;i<volume.end;++i) {
                        const auto& g=s.gripper[i];
                        const Eigen::Vector3f center=rotation*Eigen::Vector3f(g[0],g[1],g[2])+translation;
                        if(index->collides(Point{center.x(),center.y(),center.z()},g[3])) {broad=true;break;}
                    }
                    if(broad && convex_collision(s,volume,surface,rotation,translation)) {
                        s.last_part=1000+volume.first;s.last_surface=surface;
                        if(surface==0) ++s.static_rejects;else ++s.moving_rejects;
                        return false;
                    }
                }
            }
        }
        s.last_part = s.last_surface = -1;
        return true;
    }
    template<size_t rake>
    static bool fkcc_attach(const vamp::collision::Environment<vamp::FloatVector<rake>>& env,
                            const ConfigurationBlock<rake>& q) noexcept { return fkcc<rake>(env, q); }
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
        const float jaw=(tcp.block<3,3>(0,0)*Eigen::Vector3f::UnitX()).dot(s.axis);
        // Match CoupledHandlePoint: its cone concerns bar/up-axis alignment,
        // with free axial twist, rather than distance from a rigid TCP pose.
        const float cone=std::acos(std::clamp(std::abs(tcp.block<3,3>(0,0).col(1).dot(s.axis)),0.F,1.F));
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
        result[2]=.15F*(s.aligned_jaws ? jaw : std::copysign(std::max(0.F,std::abs(jaw)-.8660254F),jaw));
        result[3]=.15F*(theta-std::clamp(theta,-.03F,s.goal_angle+.03F));
        result[4]=.15F*std::max(0.F,cone-float(M_PI/2));
        return result;
    }
    vamp::FloatVector<lanes,1> distanceToConstraint(const Block& q) const {
        alignas(32) std::array<float,lanes> result{};
        for(size_t lane=0;lane<lanes;++lane) {
            std::array<float,7> scalar;
            for(size_t j=0;j<7;++j) scalar[j]=q[{j,lane}];
            result[lane]=error(scalar).norm();
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
            if(residual.norm()<=1e-6F) continue;
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
    s->gripper_convex.clear();
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
        s->point_grids={std::make_shared<PointGrid>(stationary,ns),
            std::make_shared<PointGrid>(moving,nm),std::make_shared<PointGrid>(contact,nc)};
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
void mcvamp_destroy(void* p) { delete static_cast<Scene*>(p); }
int mcvamp_set_grasp_constraint(void* p,int mode) {
    if(mode<0 || mode>2) return 0;
    static_cast<Scene*>(p)->grasp_constraint=mode;return 1;
}
int mcvamp_set_gripper_convex(void* p,const float* planes,const int* offsets,const float* bounds,int count,float padding) {
    try {
        auto& s=*static_cast<Scene*>(p);s.gripper_convex.clear();s.gripper_point_padding=padding;
        if(!std::isfinite(padding) || padding<0) return 0;
        for(int i=0;i<count;++i) {
            ConvexGripper v;
            for(int j=offsets[i];j<offsets[i+1];++j)
                v.planes.emplace_back(Eigen::Map<const Eigen::Vector4f>(planes+4*j));
            v.center=Eigen::Map<const Eigen::Vector3f>(bounds+7*i);
            v.radius=bounds[7*i+3];v.finger=bounds[7*i+4]>.5F;
            v.first=int(bounds[7*i+5]);v.end=int(bounds[7*i+6]);
            if(v.planes.empty() || v.first<0 || v.end>int(s.gripper.size()) || v.radius<=0) return 0;
            s.gripper_convex.push_back(std::move(v));
        }
        return count>0;
    } catch(...) {return 0;}
}
int mcvamp_set_arm_convex(void* p,const float* planes,const int* offsets,const float* bounds,int count,
                           const float* axes,const float* anchors,const float* reference) {
    try {
        auto& s=*static_cast<Scene*>(p);s.arm_convex.clear();
        for(int j=0;j<7;++j) {
            s.joint_axes[j]=Eigen::Map<const Eigen::Vector3f>(axes+3*j);
            s.joint_anchors[j]=Eigen::Map<const Eigen::Vector3f>(anchors+3*j);
            s.calibration_q[j]=reference[j];
            if(!s.joint_axes[j].allFinite() || std::abs(s.joint_axes[j].norm()-1.F)>.001F) return 0;
        }
        for(int i=0;i<count;++i) {
            ConvexGripper v;
            for(int j=offsets[i];j<offsets[i+1];++j) v.planes.emplace_back(Eigen::Map<const Eigen::Vector4f>(planes+4*j));
            v.center=Eigen::Map<const Eigen::Vector3f>(bounds+5*i);v.radius=bounds[5*i+3];v.first=int(bounds[5*i+4]);
            if(v.planes.empty() || v.first<1 || v.first>7 || v.radius<=0) return 0;
            s.arm_convex.push_back(std::move(v));
        }
        return count>=6;
    } catch(...) {return 0;}
}
int mcvamp_set_existing_palm_tolerance(void* p,float tolerance) {
    if(!std::isfinite(tolerance) || tolerance<0.F || tolerance>.0005F) return 0;
    static_cast<Scene*>(p)->existing_palm_tolerance=tolerance;return 1;
}
int mcvamp_validate(void* p,const float* q,double* stats) {
    auto& s=*static_cast<Scene*>(p); active=&s; s.deadline=Clock::now()+std::chrono::seconds(60);
    const vamp::collision::Environment<vamp::FloatVector<lanes>> empty;
    const bool ok=ObservedPanda::fkcc<lanes>(empty,block(q));
    statistics(s,stats); return ok;
}
// Project and test candidates during planning; never called by the executor.
int mcvamp_project(void* p,const float* tsr,const float* q,float* result,double* residual) {
    auto& s=*static_cast<Scene*>(p); active=&s;
    auto c=constraint(tsr); auto b=block(q), projected=b;
    c->projectConfiguration(b,projected,vamp::planning::constraint::InnerLM,.5F,1.F,100,false);
    for(size_t j=0;j<7;++j) result[j]=projected[{j,0}];
    *residual=c->distanceToConstraint(projected)[{0,0}];
    return std::isfinite(*residual) && *residual<=1e-6;
}
// Solve and perform constraint-aware dense resampling before returning a plan.
// stats[8:]: solve seconds, resample seconds, iterations, sparse count, dense count.
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
            solved=vamp::planning::RRTC<ObservedPanda,lanes,32>::solve(a,b,empty,settings,rng);
        }
        const auto resample=Clock::now();
        stats[8]=std::chrono::duration<double>(resample-before).count();
        stats[10]=solved.iterations; stats[11]=solved.path.size(); stats[12]=0;
        if(solved.path.size()<2) { statistics(s,stats); return 0; }
        int n=0;
        alignas(32) std::array<float,8> qa{},qb{};
        // Resampling is a planning operation. Reject invalid dense solutions
        // before execution; the executor subsequently sends every reference.
        for(size_t k=0;k+1<solved.path.size();++k) {
            solved.path[k].to_array(qa.data());solved.path[k+1].to_array(qb.data());
            float gap=0;
            for(size_t j=0;j<7;++j) gap=std::max(gap,std::abs(qb[j]-qa[j]));
            const int subdivisions=std::max(1,int(std::ceil(gap/.006F)));
            for(int i=0;i<subdivisions;++i) {
                if(n>=capacity) return -2;
                std::array<float,7> raw{};
                for(size_t j=0;j<7;++j) raw[j]=qa[j]+float(i)/subdivisions*(qb[j]-qa[j]);
                auto x=block(raw.data()), projected=x;
                if(c) {
                    c->projectConfiguration(x,projected,vamp::planning::constraint::InnerLM,.2F,1.F,100,false);
                    if(c->distanceToConstraint(projected)[{0,0}]>1e-6F) { statistics(s,stats); return -3; }
                }
                if(!ObservedPanda::fkcc<lanes>(empty,projected)) { statistics(s,stats); return -4; }
                for(size_t j=0;j<7;++j) output[n*7+j]=projected[{j,0}];
                ++n;
            }
        }
        if(n>=capacity) return -2;
        if(!ObservedPanda::fkcc<lanes>(empty,block(goal))) { statistics(s,stats); return -5; }
        std::copy_n(goal,7,output+n*7); ++n;
        stats[9]=std::chrono::duration<double>(Clock::now()-resample).count();
        stats[12]=n; statistics(s,stats); return n;
    } catch (...) { return -9; }
}
}
