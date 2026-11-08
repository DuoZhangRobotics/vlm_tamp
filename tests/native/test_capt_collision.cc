// The collision tree must agree with point/sphere distance, including cross-cell neighbors.
#include <vamp/collision/capt.hh>

#include <array>
#include <cmath>
#include <iostream>
#include <random>
#include <vector>

using vamp::collision::CAPT;
using vamp::collision::Point;
using Vector = vamp::FloatVector<>;

bool brute_collision(const std::vector<Point>& points, const Point& center, float radius) {
    for (const auto& point : points) {
        float distance = 0.F;
        for (int axis = 0; axis < 3; ++axis) {
            const float delta = point[axis] - center[axis];
            distance += delta * delta;
        }
        if (distance <= radius * radius) return true;
    }
    return false;
}

struct Counts {
    int queries = 0;
    int scalar_mismatches = 0;
    int simd_mismatches = 0;
};

void check_query(CAPT& tree, const std::vector<Point>& points, const Point& center,
                 float radius, Counts& counts) {
    const bool expected = brute_collision(points, center, radius);
    const bool scalar = tree.collides(center, radius);
    const std::array<Vector, 3> batch = {
        Vector::fill(center[0]), Vector::fill(center[1]), Vector::fill(center[2])};
    const bool simd = tree.collides_simd(batch, Vector::fill(radius));
    ++counts.queries;
    counts.scalar_mismatches += scalar != expected;
    counts.simd_mismatches += simd != expected;
}

int main() {
    Counts counts;
    // The nearest obstacle is in the lower partition, close to the split.
    // A far point at the beginning of that sorted half must not hide it.
    const std::vector<Point> adversarial = {
        Point{0.F, 0.F, 0.F}, Point{.5F, 0.F, 0.F},
        Point{.6F, 1.F, 0.F}, Point{1.F, 1.F, 0.F}};
    CAPT first(adversarial, .012F, .08F, 0.F);
    const Point upper_query{.555F, 0.F, 0.F};
    const bool minimal_expected = brute_collision(adversarial, upper_query, .06F);
    const bool minimal_actual = first.collides(upper_query, .06F);
    check_query(first, adversarial, upper_query, .06F, counts);
    check_query(first, adversarial, Point{.545F, 1.F, 0.F}, .06F, counts);
    check_query(first, adversarial, Point{2.F, 2.F, 2.F}, .08F, counts);
    check_query(first, adversarial, Point{.505F, 0.F, 0.F}, .012F, counts);

    std::mt19937 generator(9342);
    std::uniform_real_distribution<float> location(-1.F, 1.F);
    std::uniform_real_distribution<float> radius(.0125F, .079F);
    for (const int size : {3, 4, 7, 17, 31, 64, 255, 257}) {
        for (const bool planar : {false, true}) {
            std::vector<Point> points;
            for (int i = 0; i < size; ++i)
                points.push_back(Point{location(generator), location(generator),
                                       planar ? 0.F : location(generator)});
            CAPT tree(points, .012F, .08F, 0.F);
            for (int i = 0; i < 256; ++i) {
                Point center{location(generator), location(generator), location(generator)};
                const float r = radius(generator);
                if (i % 2 == 0) {
                    const Point& target = points[i % points.size()];
                    const float norm = std::sqrt(center[0] * center[0] + center[1] * center[1] +
                                                 center[2] * center[2]);
                    for (int axis = 0; axis < 3; ++axis)
                        center[axis] = target[axis] + center[axis] * (.8F * r / norm);
                }
                check_query(tree, points, center, r, counts);
            }
        }
    }
    std::cout << "{\"queries\":" << counts.queries
              << ",\"minimal_expected\":" << (minimal_expected ? "true" : "false")
              << ",\"minimal_actual\":" << (minimal_actual ? "true" : "false")
              << ",\"scalar_mismatches\":" << counts.scalar_mismatches
              << ",\"simd_mismatches\":" << counts.simd_mismatches << "}\n";
    return counts.scalar_mismatches || counts.simd_mismatches ? 1 : 0;
}
