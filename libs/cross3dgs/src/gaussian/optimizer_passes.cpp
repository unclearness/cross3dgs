#include "program_impl.h"
#include "shaders.h"

#include <utility>

namespace cross3dgs {
using namespace detail;

void GaussianForwardProgram::Impl::addOptimizerPasses(bool fuse_projection_optimizer) {
    if (!fuse_projection_optimizer) {
        // Multi-step Update zeroes the grad buffer inside the Adam pass
        // when it steps, so the gradient binding must be writable.
        const bool mu = config.igs_multistep_update;
        const uint32_t mu_start = config.igs_refine_stop;
        const uint32_t mu_stage2 = config.igs_mu_stage2_step;
        const uint32_t mu_n1 = config.igs_mu_n1;
        const uint32_t mu_n2 = config.igs_mu_n2;
        const auto grad_access =
                mu ? gpu::AccessMode::ReadWrite : gpu::AccessMode::Read;
        const auto add_adam = [&](const std::string& label, std::string shader,
                                  const gpu::BufferVariable& parameter,
                                  const gpu::BufferVariable& gradient,
                                  const gpu::BufferVariable& moment,
                                  const gpu::BufferVariable& variance,
                                  size_t element_capacity) {
            program.addPass({label,
                             std::move(shader),
                             {parameter, gradient, moment, variance, optimizer_meta},
                             {{parameter,
                               gpu::PipelineStage::Compute,
                               gpu::AccessMode::ReadWrite,
                               {}},
                              {gradient, gpu::PipelineStage::Compute, grad_access, {}},
                              {moment,
                               gpu::PipelineStage::Compute,
                               gpu::AccessMode::ReadWrite,
                               {}},
                              {variance,
                               gpu::PipelineStage::Compute,
                               gpu::AccessMode::ReadWrite,
                               {}},
                              {optimizer_meta,
                               gpu::PipelineStage::Compute,
                               gpu::AccessMode::Read,
                               {}}},
                             gpu::DispatchDesc::Direct(static_cast<uint32_t>(
                                     (element_capacity + 255) / 256))});
        };
        add_adam("adam_means",
                 AdamShader(0, config.means_lr, 0.0f, 0.0f, 0.0f, config.means_lr_final,
                            config.max_steps, mu, mu_start, mu_stage2, mu_n1, mu_n2,
                            config.fp32_adam_moments),
                 means, means_grad, means_m, means_v, config.gaussian_capacity);
        add_adam("adam_scale_opacity",
                 AdamShader(1, config.scales_lr, config.opacity_lr,
                            config.scale_regularizer, config.opacity_regularizer,
                            config.scales_lr_final, config.max_steps, mu, mu_start,
                            mu_stage2, mu_n1, mu_n2, config.fp32_adam_moments),
                 scales_opacity, scale_grad, scale_m, scale_v,
                 config.gaussian_capacity);
        add_adam("adam_rotation",
                 AdamShader(2, config.rotation_lr, config.rotation_lr, 0.0f, 0.0f, 0.0f,
                            config.max_steps, mu, mu_start, mu_stage2, mu_n1, mu_n2,
                            config.fp32_adam_moments),
                 rotations, rotation_grad, rotation_m, rotation_v,
                 config.gaussian_capacity);
        add_adam("adam_sh_dc",
                 AdamShader(3, config.sh_dc_lr, 0.0f, 0.0f, 0.0f, 0.0f,
                            config.max_steps, mu, mu_start, mu_stage2, mu_n1, mu_n2,
                            config.fp32_adam_moments),
                 sh_dc, sh_grad, sh_m, sh_v, config.gaussian_capacity);
        const auto sh_adam_shader = [&]() {
            return AdamShader(4, config.sh_rest_lr, config.sh_rest_lr, 0.0f, 0.0f, 0.0f,
                              config.max_steps, mu, mu_start, mu_stage2, mu_n1, mu_n2,
                              config.fp32_adam_moments);
        };
        add_adam(split_sh_rest_storage ? "adam_sh_rest_a" : "adam_sh_rest",
                 sh_adam_shader(), sh_rest, sh_rest_grad, sh_rest_m, sh_rest_v,
                 config.gaussian_capacity * (split_sh_rest_storage ? 6u : 12u));
        if (split_sh_rest_storage) {
            add_adam("adam_sh_rest_b", sh_adam_shader(), sh_rest_b, sh_rest_grad_b,
                     sh_rest_m_b, sh_rest_v_b, config.gaussian_capacity * 6u);
        }
    }
}

}  // namespace cross3dgs
