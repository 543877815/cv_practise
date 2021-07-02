import torch
from super_resolution.models.GLEAN.stylegan import G_synthesis, G_mapping
from torchvision import utils as vutils


def getNoise(seed=0, batch_size=2):
    seed = seed
    batch_size = batch_size
    with torch.no_grad():
        torch.manual_seed(seed)
        latent = torch.rand((batch_size, 512), dtype=torch.float32, device="cuda")
        latent_out = torch.nn.LeakyReLU(5)(mapping(latent))
        gaussian_fit = {"mean": latent_out.mean(0), "std": latent_out.std(0)}
        torch.save(gaussian_fit, "gaussian_fit.pt")
        print("\tSaved \"gaussian_fit.pt\"")
    lrelu = torch.nn.LeakyReLU(negative_slope=0.2)
    latent = torch.randn(
        (batch_size, 18, 512), dtype=torch.float, requires_grad=False, device='cuda')
    latent_in = latent.expand(-1, 18, -1)
    # Apply learned linear mapping to match latent distribution to that of the mapping network
    latent_in = lrelu(latent_in * gaussian_fit["std"] + gaussian_fit["mean"])
    # Generate list of noise tensors
    noise = []  # stores all of the noise tensors
    noise_vars = []  # stores the noise tensors that we want to optimize on
    noise_type = 'fixed'
    num_trainable_noise_layers = 0
    for i in range(18):
        # dimension of the ith noise tensor
        res = (batch_size, 1, 2 ** (i // 2 + 2), 2 ** (i // 2 + 2))
        if noise_type == 'fixed':
            new_noise = torch.randn(res, dtype=torch.float, device='cuda')
            new_noise.requires_grad = False
        elif noise_type == 'trainable':
            new_noise = torch.randn(res, dtype=torch.float, device='cuda')
            if i < num_trainable_noise_layers:
                new_noise.requires_grad = True
                noise_vars.append(new_noise)
            else:
                new_noise.requires_grad = False
        else:
            raise Exception("unknown noise type")
        noise.append(new_noise)
    return latent_in, noise


def swapNoise(noise1, noise2, level):
    first = level * 2
    tmp = noise1.clone()
    noise1[:, first, :] = noise2[:, first, :]
    noise2[:, first, :] = tmp[:, first, :]

    second = level * 2 + 1
    noise1[:, second, :] = noise2[:, second, :]
    noise2[:, second, :] = tmp[:, second, :]

    return noise1, noise2


if __name__ == '__main__':
    synthesis = G_synthesis().cuda()
    mapping = G_mapping().cuda()

    synthesis.load_state_dict(torch.load("pretrained_model/synthesis.pt"))
    mapping.load_state_dict(torch.load("pretrained_model/mapping.pt"))

    print("Running Mapping Network!")
    with torch.no_grad():
        seed1 = 7
        latent_in1, noise1 = getNoise(seed=seed1, batch_size=2)
        gen_im = (synthesis(latent_in1, noise1) + 1) / 2
        vutils.save_image(gen_im, 'images/{}.png'.format(seed1))

        seed2 = 8
        latent_in2, noise2 = getNoise(seed=seed2, batch_size=2)
        gen_im = (synthesis(latent_in2, noise2) + 1) / 2
        vutils.save_image(gen_im, 'images/{}.png'.format(seed2))

        latent_in1, latent_in2 = swapNoise(latent_in1, latent_in2, level=0)
        latent_in1, latent_in2 = swapNoise(latent_in1, latent_in2, level=1)
        new_in1, new_in2 = swapNoise(latent_in1, latent_in2, level=2)

        gen_im1 = (synthesis(new_in1, noise1) + 1) / 2
        vutils.save_image(gen_im1, 'images/{}_{}.png'.format(seed1, seed2))

        gen_im2 = (synthesis(new_in2, noise2) + 1) / 2
        vutils.save_image(gen_im2, 'images/{}_{}.png'.format(seed2, seed1))
