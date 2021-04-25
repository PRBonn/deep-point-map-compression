#include "OctreeHandler.h"

#include <pybind11/eigen.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <eigen3/Eigen/Dense>
#include <vector>

namespace py = pybind11;

void Octree::setInput(Eigen::MatrixXf &cloud) {
  points_.clear();
  points_.reserve(cloud.rows());
  for (uint32_t i = 0; i < cloud.rows(); i++) {
    points_.emplace_back(
        Eigen::Vector3f(cloud(i, 0), cloud(i, 1), cloud(i, 2)));
  }
  octree_.initialize(points_);
}

std::vector<uint32_t> Octree::radiusSearch(const Eigen::Vector3f &pt,
                                           const float &radius) {
  std::vector<uint32_t> results;
  octree_.radiusNeighbors<unibn::L2Distance<Eigen::Vector3f>>(
      pt, radius, results);
  return results;
}

std::vector<uint32_t> Octree::radiusSearch(const uint32_t &pt_idx,
                                           const float &radius) {
  assert(pt_idx < points_.size());
  std::vector<uint32_t> results;
  const Eigen::Vector3f &q = points_[pt_idx];
  octree_.radiusNeighbors<unibn::L2Distance<Eigen::Vector3f>>(
      q, radius, results);
  return results;
}

Eigen::MatrixXi Octree::radiusSearchAll(const uint32_t &max_nr_neighbors,
                                        const float &radius) {
  uint32_t act_max_neighbors = 0;
  Eigen::MatrixXi indices =
      Eigen::MatrixXi::Ones(points_.size(), max_nr_neighbors) *
      (int)points_.size();
  for (uint32_t i = 0; i < points_.size(); i++) {
    std::vector<uint32_t> results = radiusSearch(i, radius);
    uint32_t nr_n = std::min(max_nr_neighbors, (uint32_t)results.size());
    if (nr_n > act_max_neighbors) act_max_neighbors = nr_n;
    for (uint32_t j = 0; j < nr_n; j++) {
      indices(i, j) = results[j];
    }
  }
  return indices.leftCols(act_max_neighbors);
}

Eigen::MatrixXi Octree::radiusSearchIndices(
    const std::vector<uint32_t> &pt_indices,
    const uint32_t &max_nr_neighbors,
    const float &radius) {
  uint32_t act_max_neighbors = 0;
  Eigen::MatrixXi indices =
      Eigen::MatrixXi::Ones(pt_indices.size(), max_nr_neighbors) *
      (int)points_.size();
  for (uint32_t i = 0; i < pt_indices.size(); i++) {
    std::vector<uint32_t> results = radiusSearch(pt_indices[i], radius);
    uint32_t nr_n = std::min(max_nr_neighbors, (uint32_t)results.size());
    if (nr_n > act_max_neighbors) act_max_neighbors = nr_n;
    for (uint32_t j = 0; j < nr_n; j++) {
      uint32_t step =
          std::max((uint32_t)(results.size() * j / max_nr_neighbors), j);
      assert(step < results.size());
      indices(i, j) = results[step];
    }
  }
  return indices.leftCols(act_max_neighbors);
}

Eigen::MatrixXi Octree::radiusSearchPoints(Eigen::MatrixXf &query_points,
                                           const uint32_t &max_nr_neighbors,
                                           const float &radius) {
  assert(query_points.cols() == 3);
  uint32_t act_max_neighbors = 0;
  Eigen::MatrixXi indices =
      Eigen::MatrixXi::Ones(query_points.rows(), max_nr_neighbors) *
      (int)points_.size();
  for (uint32_t i = 0; i < query_points.rows(); i++) {
    std::vector<uint32_t> results = radiusSearch(query_points.row(i), radius);
    uint32_t nr_n = std::min(max_nr_neighbors, (uint32_t)results.size());
    if (nr_n > act_max_neighbors) act_max_neighbors = nr_n;
    for (uint32_t j = 0; j < nr_n; j++) {
      uint32_t step =
          std::max((uint32_t)(results.size() * j / max_nr_neighbors), j);
      assert(step < results.size());
      assert(results[step] <= points_.size());
      // std::cout<<results[step]<<" "<<std::endl;
      indices(i, j) = results[step];
    }
  }
  return indices.leftCols(act_max_neighbors);
}

std::vector<uint32_t> Octree::randomUniformSampling(const float &max_dist) {
  std::vector<uint32_t> indices;
  if (max_dist > 0) {
    octree_.randomUniformSampling(max_dist, indices);
  } else {
    indices = std::vector<uint32_t>(points_.size());  // return all points
    std::iota(std::begin(indices), std::end(indices), 0);
  }
  return indices;
}
Eigen::MatrixXf Octree::computeScatterMatrices(const float &radius) {
  uint32_t n = points_.size();
  Eigen::MatrixXf scatter_matrices = Eigen::MatrixXf(n, 6);
  for (uint32_t i = 0; i < n; i++) {
    std::vector<uint32_t> neighbors_idx = radiusSearch(i, radius);
    uint32_t nr_neighbors = neighbors_idx.size();
    Eigen::Matrix3Xf neighbors = Eigen::Matrix3Xf(3, nr_neighbors);
    for (uint32_t j = 0; j < nr_neighbors; j++) {
      neighbors.col(j) = points_[neighbors_idx[j]];
    }
    // https://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
    Eigen::Matrix3Xf centered =
        (neighbors.colwise() - neighbors.rowwise().mean()) / radius;
    Eigen::Matrix3f cov =
        (centered * centered.transpose()) / float(nr_neighbors - 1);
    Eigen::Matrix<float, 1, 6> flat_cov;
    flat_cov << cov(0, 0), cov(1, 1), cov(2, 2), cov(0, 1), cov(0, 2),
        cov(1, 2);
    scatter_matrices.row(i) = flat_cov;
  }
  return scatter_matrices;
}

Eigen::MatrixXf Octree::spectralFeaturesAll(const float &radius) {
  uint32_t n = points_.size();
  Eigen::MatrixXf eigenvalues = Eigen::MatrixXf(n, 3);
  for (uint32_t i = 0; i < n; i++) {
    std::vector<uint32_t> neighbors_idx = radiusSearch(i, radius);
    uint32_t nr_neighbors = neighbors_idx.size();
    Eigen::Matrix3Xf neighbors = Eigen::Matrix3Xf(3, nr_neighbors);
    for (uint32_t j = 0; j < nr_neighbors; j++) {
      neighbors.col(j) = points_[neighbors_idx[j]];
    }
    // https://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
    Eigen::Matrix3Xf centered =
        (neighbors.colwise() - neighbors.rowwise().mean()) / radius;
    Eigen::Matrix3f cov =
        (centered * centered.transpose()) / float(nr_neighbors - 1);
    Eigen::Vector3f eigenv = cov.eigenvalues().real();
    std::sort(
        eigenv.data(), eigenv.data() + eigenv.size(), std::greater<float>());
    eigenvalues.row(i) = eigenv.transpose();
  }
  return eigenvalues;
}
Eigen::MatrixXf Octree::computeEigenvaluesNormal(const float &radius) {
  uint32_t n = points_.size();
  Eigen::MatrixXf eigenvalues = Eigen::MatrixXf(n, 6);
  for (uint32_t i = 0; i < n; i++) {
    std::vector<uint32_t> neighbors_idx = radiusSearch(i, radius);
    uint32_t nr_neighbors = neighbors_idx.size();
    Eigen::Matrix3Xf neighbors = Eigen::Matrix3Xf(3, nr_neighbors);
    for (uint32_t j = 0; j < nr_neighbors; j++) {
      neighbors.col(j) = points_[neighbors_idx[j]];
    }
    // https://stackoverflow.com/questions/15138634/eigen-is-there-an-inbuilt-way-to-calculate-sample-covariance
    Eigen::Matrix3Xf centered =
        (neighbors.colwise() - neighbors.rowwise().mean()) / radius;
    Eigen::Matrix3f cov =
        (centered * centered.transpose()) / float(nr_neighbors - 1);
    Eigen::JacobiSVD<Eigen::Matrix3f> svd(cov, Eigen::ComputeFullU);
    Eigen::Matrix3f eigenvectors = svd.matrixU();
    // last column is the eigenvector of the smalles singular value
    Eigen::Vector3f normal = eigenvectors.col(2);
    normal.normalize();
    Eigen::Vector3f singularv = svd.singularValues();
    if (i < 1) {
      std::cout << "sing: " << singularv
                << "sqrt sing: " << singularv.array().sqrt().real()
                << "pow sing: " << singularv.array().pow(2) << "eigen "
                << cov.eigenvalues().real() << std::endl;
    }
    eigenvalues.row(i).leftCols(3) = singularv.transpose();
    eigenvalues.row(i).rightCols(3) = normal.transpose();
  }
  return eigenvalues;
}

// if fatal error: Python.h: No such file or directory
// https://github.com/stevenlovegrove/Pangolin/issues/494
// CPLUS_INCLUDE_PATH=/usr/include/python3.6
// export CPLUS_INCLUDE_PATH
PYBIND11_MODULE(octree_handler, m) {
  m.doc() = "pybind11 octree plugin";  // optional module docstring

  m.def("scale",
        [](pybind11::EigenDRef<Eigen::MatrixXd> m, double c) { m *= c; });
  py::class_<Octree>(m, "Octree")
      .def(py::init())
      .def("setInput",
           &Octree::setInput,
           "builds octree based on the input cloud",
           py::arg("points"))
      .def("radiusSearch",
           py::overload_cast<const uint32_t &, const float &>(
               &Octree::radiusSearch))
      .def("radiusSearchAll", &Octree::radiusSearchAll)
      .def("radiusSearchIndices", &Octree::radiusSearchIndices)
      .def("radiusSearchPoints", &Octree::radiusSearchPoints)
      .def("spectralFeatureAll", &Octree::spectralFeaturesAll)
      .def("computeEigenvaluesNormal", &Octree::computeEigenvaluesNormal)
      .def("computeScatterMatrices", &Octree::computeScatterMatrices)
      .def("randomUniformSampling", &Octree::randomUniformSampling);
}
