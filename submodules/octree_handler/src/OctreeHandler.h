#pragma once

#include <Eigen/Core>
#include <vector>

#include "Octree.hpp"

namespace unibn {
namespace traits {

template <>
struct access<Eigen::Vector3f, 0> {
  static float get(const Eigen::Vector3f &p) { return p.x(); }
};

template <>
struct access<Eigen::Vector3f, 1> {
  static float get(const Eigen::Vector3f &p) { return p.y(); }
};

template <>
struct access<Eigen::Vector3f, 2> {
  static float get(const Eigen::Vector3f &p) { return p.z(); }
};
}  // namespace traits
}  // namespace unibn

class Octree {
 private:
  unibn::Octree<Eigen::Vector3f> octree_;
  std::vector<Eigen::Vector3f> points_;

 public:
  Octree() = default;
  ~Octree() = default;
  void setInput(Eigen::MatrixXf &cloud);
  std::vector<uint32_t> radiusSearch(const uint32_t &pt_idx,
                                     const float &radius);
  std::vector<uint32_t> radiusSearch(const Eigen::Vector3f &pt,
                                     const float &radius);
  Eigen::MatrixXi radiusSearchAll(const uint32_t &max_nr_neighbors,
                                  const float &radius);
  Eigen::MatrixXi radiusSearchIndices(const std::vector<uint32_t> &pt_indices,
                                      const uint32_t &max_nr_neighbors,
                                      const float &radius);
  Eigen::MatrixXi radiusSearchPoints(Eigen::MatrixXf &query_points,
                                     const uint32_t &max_nr_neighbors,
                                     const float &radius);
  std::vector<uint32_t> randomUniformSampling(const float &max_dist);
  Eigen::MatrixXf spectralFeaturesAll(const float &radius);
  Eigen::MatrixXf computeEigenvaluesNormal(const float &radius);
  Eigen::MatrixXf computeScatterMatrices(const float &radius);
};
