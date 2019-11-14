#ifndef VGG_H
#define VGG_H

#include <torch/torch.h>
#include "general.h"

namespace models {
	struct VISION_API VGGImpl: torch::nn::Module {
		torch::nn::Sequential features{ nullptr }, classifier{ nullptr };

		VGGImpl(torch::nn::Sequential features, long long num_classes, bool initialize_weights = true);

		torch::Tensor forward(torch::Tensor x);
		void _initialize_weights();
	};

	struct VISION_API VGG11Impl: VGGImpl {
		VGG11Impl(long long num_classes = 1000, bool initialize_weights = true);
	};

	struct VISION_API VGG13Impl: VGGImpl {
		VGG13Impl(long long num_classes = 1000, bool initialize_weights = true);
	};

	struct VISION_API VGG16Impl: VGGImpl {
		VGG16Impl(long long num_classes = 1000, bool initialize_weights = true);
	};

	struct VISION_API VGG19Impl: VGGImpl {
		VGG19Impl(long long num_classes = 1000, bool initialize_weights = true);
	};

	struct VISION_API VGG11BNImpl: VGGImpl {
		VGG11BNImpl(long long num_classes = 1000, bool initialize_weights = true);
	};

	struct VISION_API VGG13BNImpl: VGGImpl {
		VGG13BNImpl(long long num_classes = 1000, bool initialize_weights = true);
	};

	struct VISION_API VGG16BNImpl: VGGImpl {
		VGG16BNImpl(long long num_classes = 1000, bool initialize_weights = true);
	};
	struct VISION_API VGG19BNImpl: VGGImpl {
		VGG19BNImpl(long long num_classes = 1000, bool initialize_weights = true);
	};

	TORCH_MODULE(VGG);

	TORCH_MODULE(VGG11);
	TORCH_MODULE(VGG13);
	TORCH_MODULE(VGG16);
	TORCH_MODULE(VGG19);

	TORCH_MODULE(VGG11BN);
	TORCH_MODULE(VGG13BN);
	TORCH_MODULE(VGG16BN);
	TORCH_MODULE(VGG19BN);
}


#endif