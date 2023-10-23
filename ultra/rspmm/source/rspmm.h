#pragma once

#include <tuple>

#include <torch/extension.h>
//#include <ATen/SparseTensorUtils.h>
#include <ATen/native/SparseTensorUtils.h>

namespace at {

using namespace at::sparse;

void rspmm_forward_check(CheckedFrom c, const TensorArg &edge_index_arg, const TensorArg &edge_type_arg,
                         const TensorArg &edge_weight_arg, const TensorArg &relation_arg, const TensorArg &input_arg);

void rspmm_backward_check(CheckedFrom c, const TensorArg &edge_index_arg, const TensorArg &edge_type_arg,
                          const TensorArg &edge_weight_arg, const TensorArg &relation_arg, const TensorArg &input_arg,
                          const TensorArg &output_arg, const TensorArg &output_grad_arg);

Tensor ind2ptr(const Tensor &index, int size);

Tensor rspmm_add_mul_forward_cpu(const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,
                                 const Tensor &relation, const Tensor &input);

std::tuple<Tensor, Tensor, Tensor> rspmm_add_mul_backward_cpu(
        const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight, const Tensor &relation,
        const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_min_mul_forward_cpu(const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,
                                 const Tensor &relation, const Tensor &input);

std::tuple<Tensor, Tensor, Tensor> rspmm_min_mul_backward_cpu(
        const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight, const Tensor &relation,
        const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_max_mul_forward_cpu(const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,
                                 const Tensor &relation, const Tensor &input);

std::tuple<Tensor, Tensor, Tensor> rspmm_max_mul_backward_cpu(
        const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight, const Tensor &relation,
        const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_add_add_forward_cpu(const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,
                                 const Tensor &relation, const Tensor &input);

std::tuple<Tensor, Tensor, Tensor> rspmm_add_add_backward_cpu(
        const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight, const Tensor &relation,
        const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_min_add_forward_cpu(const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,
                                 const Tensor &relation, const Tensor &input);

std::tuple<Tensor, Tensor, Tensor> rspmm_min_add_backward_cpu(
        const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight, const Tensor &relation,
        const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_max_add_forward_cpu(const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,
                                 const Tensor &relation, const Tensor &input);

std::tuple<Tensor, Tensor, Tensor> rspmm_max_add_backward_cpu(
        const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight, const Tensor &relation,
        const Tensor &input, const Tensor &output, const Tensor &output_grad);

#ifdef CUDA_OP
Tensor rspmm_add_mul_forward_cuda(const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,
                                 const Tensor &relation, const Tensor &input);

std::tuple<Tensor, Tensor, Tensor> rspmm_add_mul_backward_cuda(
        const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight, const Tensor &relation,
        const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_min_mul_forward_cuda(const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,
                                 const Tensor &relation, const Tensor &input);

std::tuple<Tensor, Tensor, Tensor> rspmm_min_mul_backward_cuda(
        const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight, const Tensor &relation,
        const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_max_mul_forward_cuda(const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,
                                 const Tensor &relation, const Tensor &input);

std::tuple<Tensor, Tensor, Tensor> rspmm_max_mul_backward_cuda(
        const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight, const Tensor &relation,
        const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_add_add_forward_cuda(const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,
                                 const Tensor &relation, const Tensor &input);

std::tuple<Tensor, Tensor, Tensor> rspmm_add_add_backward_cuda(
        const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight, const Tensor &relation,
        const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_min_add_forward_cuda(const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,
                                 const Tensor &relation, const Tensor &input);

std::tuple<Tensor, Tensor, Tensor> rspmm_min_add_backward_cuda(
        const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight, const Tensor &relation,
        const Tensor &input, const Tensor &output, const Tensor &output_grad);

Tensor rspmm_max_add_forward_cuda(const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight,
                                 const Tensor &relation, const Tensor &input);

std::tuple<Tensor, Tensor, Tensor> rspmm_max_add_backward_cuda(
        const Tensor &edge_index, const Tensor &edge_type, const Tensor &edge_weight, const Tensor &relation,
        const Tensor &input, const Tensor &output, const Tensor &output_grad);
#endif

} // namespace at