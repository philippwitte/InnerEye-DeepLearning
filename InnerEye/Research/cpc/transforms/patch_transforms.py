from torchvision.transforms import ToTensor, ToPILImage, Normalize


class Patchifier:
    def __init__(self, patch_sizes, patch_strides, squeeze_output_patches=False):
        """
        Extracts sliding window patches from a image Tensors of arbitrary dim
        Patches will overlap by a factor stride // kernel_size in each dim

        kernel_size: list of ints
            Size of squared patches to extract in each dim
        stride: list of int
            Stride of sliding patch window in each dim
        squeeze_output_patches: bool
            Squeeze any dimensions of size 1 in the output patches. Note, this
            does not squeeze dimensions of size 1 in the patch grid.
        """
        self.kernel_sizes = list(map(int, patch_sizes))
        self.strides = list(map(int, patch_strides))
        self.squeeze_output_patches = bool(squeeze_output_patches)

    def __call__(self, sample):
        """
        Takes a single image and returns all sliding windows of width
        self.kernel_size and overlap self.stride along 2D or 3D spatial
        dimensions.

        sample: Tensor,
            Patched image, shape [C, ...], where ... is a number of spatial
            dimensions equal to len(self.patch_sizes).

        Returns: Tensor, patches image,
            shape [..., C, ...]
        """
        n_spatial_dims = len(self.kernel_sizes)
        if sample.ndim - 1 != n_spatial_dims:
            raise ValueError("Expected input with {} spatial dimensions given "
                             "self.patch_sizes {}, got {}".format(
                n_spatial_dims, self.kernel_sizes, sample.ndim
            ))
        for i, (kernel_size, stride) in enumerate(zip(self.kernel_sizes,
                                                      self.strides)):
            sample = sample.unfold(i+1, kernel_size, stride)
        if self.squeeze_output_patches:
            for i in range(1, n_spatial_dims+1):
                sample = sample.squeeze(-i)
        # Move channel axis in-between patch grid- and patch spatial dimensions
        axes = list(range(sample.ndim))
        axes.insert(n_spatial_dims, axes.pop(0))
        return sample.permute(*axes)


class ApplyToPatches:
    def __init__(self, transform, convert_to_pil=True):
        """
        Takes a torchvision.transform Transform object and applies it to each
        patch as output by Patchifier.

        Note: Converting back and forth from Tenosr <-> PIL.Image is slow. Try
              not to wrap individual transforms with ApplyToPatches. Instead,
              batch mulitple transforms that may act on the same datatype
              (Tensor / PIl.Image) with transforms.Compose,
              transforms.RandomApply, transforms.RandomChoice etc.

        :param transform: A torchvision.transform Transform object
        param: convert_to_pil: Bool, convert each patch to a PIL.Image object before applying the transformation.
                               Converts back to tensor afterwards if True.
        """
        self.transform = transform
        self.convert_to_pil = convert_to_pil
        self.to_img = ToPILImage()
        self.to_tensor = ToTensor()

    def __str__(self):
        return "ApplyToPatches(\n{}\n)".format(self.transform)

    @staticmethod
    def _iter_patches_2d(patches):
        for row in range(patches.shape[0]):
            for col in range(patches.shape[1]):
                yield patches[row, col], (row, col)

    @staticmethod
    def _iter_patches_3d(patches):
        for row in range(patches.shape[0]):
            for col in range(patches.shape[1]):
                for depth in range(patches.shape[2]):
                    yield patches[row, col, depth], (row, col, depth)

    def __call__(self, patches):
        """
        Iterates all patches of an input torch.Tensor of shape [g1, g2, g3, C, d1, d2, d3] (3D patches) or
        [g1, g2, C, d1, d2] (2D patches) along the dimensions g* and applies self.transform to each.

        The transform must operate on [C, d1, d2 (d3)] inputs and must not change the dimensionality of the input.

        :param patches: Transformed patches of shape [g1, g2, (g3), C, d1, d2, (d3)]
        :return:
        """
        if patches.ndim == 5:
            # 2D images and patches
            iter_func = self._iter_patches_2d
        elif patches.ndim == 7:
            # 3D images and patches
            iter_func = self._iter_patches_3d
        else:
            raise NotImplementedError("Only implemented for 2D/3D images and "
                                      "patches. Expected input to have ndim 5 "
                                      "or 7, got {}".format(patches.ndim))
        for patch, indices in iter_func(patches):
            if self.convert_to_pil:
                patch = self.to_img(patch)
            patch = self.transform(patch)
            if self.convert_to_pil:
                patch = self.to_tensor(patch)
            patches[indices] = patch
        return patches


class NormalizePatches(Normalize):
    def __call__(self, patches):
        """
        Reshapes a patched input to image-like and invokes torchvision
        transforms.Normalize on it, then reshapes and return as patches
        Parameters
        ----------
        patches: Tensor, shape [Gh, Gw, C, H, W]

        Returns
        -------
        Normalized Tensor, shape [Gh, Gw, C, H, W]
        """
        reshaped = patches.permute(2, 0, 1, 3, 4)
        shape_mem = reshaped.shape
        reshaped = reshaped.reshape(
            patches.shape[2], patches.shape[0]*patches.shape[3], -1
        )
        reshaped = super(NormalizePatches, self).__call__(reshaped)
        reshaped = reshaped.reshape(shape_mem).permute(1, 2, 0, 3, 4)
        return reshaped
