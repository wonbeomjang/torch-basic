#ifndef VGG_H
#define VGG_H

#include <torch/torch.h>

namespace models {
	struct VGGImpl : torch::nn::Module {
		torch::nn::Sequential features{ nullptr }, classifier{ nullptr };

		VGGImple(torch::nn::Sequential features, int64_t num_classes, bool initialize_weights = true);

		forward(torch::Tensor x);
		_initialize_weights();
	};

	// VGG 11-layer model (configuration "A")
	struct VGG11Impl : VGGImpl {
		VGG11Impl(int64_t num_classes = 1000, bool initialize_weights = true);
	};

	// VGG 13-layer model (configuration "B")
	struct VGG13Impl : VGGImpl {
		VGG13Impl(int64_t num_classes = 1000, bool initialize_weights = true);
	};

	// VGG 16-layer model (configuration "D")
	struct VGG16Impl : VGGImpl {
		VGG16Impl(int64_t num_classes = 1000, bool initialize_weights = true);
	};

	// VGG 19-layer model (configuration "E")
	struct VGG19Impl : VGGImpl {
		VGG19Impl(int64_t num_classes = 1000, bool initialize_weights = true);
	};

	// VGG 11-layer model (configuration "A") with batch normalization
	struct VGG11BNImpl : VGGImpl {
		VGG11BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
	};

	// VGG 13-layer model (configuration "B") with batch normalization
	struct VGG13BNImpl : VGGImpl {
		VGG13BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
	};

	// VGG 16-layer model (configuration "D") with batch normalization
	struct VGG16BNImpl : VGGImpl {
		VGG16BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
	};

	// VGG 19-layer model (configuration 'E') with batch normalization
	struct VGG19BNImpl : VGGImpl {
		VGG19BNImpl(int64_t num_classes = 1000, bool initialize_weights = true);
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