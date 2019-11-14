#include "vgg.h"
#include "modelsimpl.h"
#include <unordered_map>

namespace models {
	static std::unordered_map<char, std::vector<int>> cfgs = {
		{'A', {64, -1, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}},
		{'B', {64, 64, -1, 128, 128, -1, 256, 256, -1, 512, 512, -1, 512, 512, -1}},
		{'D', {64, 64, -1, 128, 128, -1, 256, 256, 256, -1, 512, 512, 512, -1, 512, 512, 512, -1}},
		{'E', {64,  64,  -1,  128, 128, -1,  256, 256, 256, 256, -1, 512, 512, 512, 512, -1,  512, 512, 512, 512, -1}}
	};

	torch::nn::Sequential makeLayers(const std::vector<int> cfg, bool batch_norm = false) {
		torch::nn::Sequential seq;

		int channel = 3;
		for (const auto& V : cfg) {
			if (V <= -1) {
				seq->push_back(torch::nn::Functional(modelsimpl::max_pool2d, 2, 2));
			}
			else {
				seq->push_back(torch::nn::Conv2d(torch::nn::Conv2dOptions(channel, V, 3).padding(1)));
				if (batch_norm)
					seq->push_back(torch::nn::BatchNorm(V));
				seq->push_back(torch::nn::Functional(modelsimpl::relu_));

				channel = V;
			}
		}

		return seq;
	}

	VGGImpl::VGGImpl(torch::nn::Sequential features, long long num_classes, bool initialize_weights) {
		this->features = features;

		classifier = torch::nn::Sequential(
			torch::nn::Linear(torch::nn::LinearOptions(512 * 7 * 7, 4096)),
			torch::nn::Functional(modelsimpl::relu_),
			torch::nn::Dropout(),
			torch::nn::Linear(torch::nn::LinearOptions(4096, 4096)),
			torch::nn::Functional(modelsimpl::relu_),
			torch::nn::Dropout(),
			torch::nn::Linear(torch::nn::LinearOptions(4096, num_classes))
		);
		
		if (initialize_weights)
			_initialize_weights();
	}

	torch::Tensor VGGImpl::forward(torch::Tensor x) {
		x = features->forward(x);
		x = torch::adaptive_avg_pool2d(x, { 7, 7 });
		x = x.view({ x.size(0), -1 });
		x = classifier->forward(x);

		return x;
	}

	void VGGImpl::_initialize_weights() {
		for (auto& module : modules(/*include_self=*/false)) {
			if (auto M = dynamic_cast<torch::nn::Conv2dImpl*>(module.get())) {
				torch::nn::init::kaiming_normal_(
					M->weight,
					0,
					torch::nn::init::FanMode::FanOut,
					torch::nn::init::Nonlinearity::ReLU);
				torch::nn::init::constant_(M->bias, 0);
			}
			else if (auto M = dynamic_cast<torch::nn::BatchNormImpl*>(module.get())) {
				torch::nn::init::constant_(M->weight, 1);
				torch::nn::init::constant_(M->bias, 0);
			}
			else if (auto M = dynamic_cast<torch::nn::LinearImpl*>(module.get())) {
				torch::nn::init::normal_(M->weight, 0, 0.01);
				torch::nn::init::constant_(M->bias, 0);
			}
		}
	}

	VGG11Impl::VGG11Impl(long long num_classes, bool initialize_weights)
		:VGGImpl(makeLayers(cfgs['A']), num_classes, initialize_weights) {}

	VGG13Impl::VGG13Impl(long long num_classes, bool initialize_weights )
		:VGGImpl(makeLayers(cfgs['B']), num_classes, initialize_weights) {}

	VGG16Impl::VGG16Impl(long long num_classes, bool initialize_weights)
		:VGGImpl(makeLayers(cfgs['C']), num_classes, initialize_weights) {}

	VGG19Impl::VGG19Impl(long long num_classes, bool initialize_weights)
		:VGGImpl(makeLayers(cfgs['D']), num_classes, initialize_weights) {}
		
	VGG11BNImpl::VGG11BNImpl(long long num_classes, bool initialize_weights)
		:VGGImpl(makeLayers(cfgs['A'], true), num_classes, initialize_weights) {}

	VGG13BNImpl::VGG13BNImpl(long long num_classes, bool initialize_weights)
		:VGGImpl(makeLayers(cfgs['B'], true), num_classes, initialize_weights) {}

	VGG16BNImpl::VGG16BNImpl(long long num_classes, bool initialize_weights)
		:VGGImpl(makeLayers(cfgs['C'], true), num_classes, initialize_weights) {}

	VGG19BNImpl::VGG19BNImpl(long long num_classes, bool initialize_weights)
		:VGGImpl(makeLayers(cfgs['D'], true), num_classes, initialize_weights) {}
}

