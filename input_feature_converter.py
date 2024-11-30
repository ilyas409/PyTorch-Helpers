class InputFeatureConverter:
    def __init__(
        self,
        device: torch.device,
        width: float,
        normalization_offset: float,
        normalization_max_value: float,
        n_channels: int = 1,
    ) -> None:
        if normalization_max_value == 0:
            raise ValueError("Normalization max value cannot be zero.")

        self._device = device
        self._width = width
        self._normalization_offset = normalization_offset
        self._normalization_max_value = normalization_max_value
        self._n_channels = n_channels
        
    def convert(self, df: pd.DataFrame) -> torch.Tensor:
        """
        Creates a normalized reshaped tensor from the dataframe
        that can be used as input to a `Conv2d` layer
        """
        return self._normalize(
            self._reshape(
                self._convert_to_float_tensor(df)
            )
        )

    def _convert_to_float_tensor(self, df: pd.DataFrame) -> torch.Tensor:
        """Converts a Pandas DataFrame to a float tensor."""
        return torch.from_numpy(df.to_numpy()).to(self._device).float()

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        """Reshapes the tensor to (N, C, W, W)."""
        n_rows = len(x)
        return x.reshape(n_rows, self._n_channels, self._width, self._width)

    def _normalize(self, x: torch.Tensor) -> torch.Tensor:
        """Normalizes the tensor to range specified by the normalization parameters."""
        return x / self._normalization_max_value - self._normalization_offset

class TensorImageDataset(Dataset):
    def __init__(
        self,
        csv_path: str,
        n_channels: int,
        device: torch.device,
        width: int,
        label_name: str = "label",
    ) -> None:
        """
        Args:
            csv_path (str): Path to the CSV file containing both data and labels.
            n_channels (int): Number of channels in the images.
            device (torch.device): Device to which tensors will be moved.
            width (int): Image width (assuming square images).
            label_name (str): Column name for labels.
        """
        self._device = device
        self._n_channels = n_channels
        self._converter = InputFeatureConverter(
            device=device,
            width=width,
            normalization_offset=0.5,
            normalization_max_value=255.0,
            n_channels=1,
        )

        df = pd.read_csv(csv_path)

        self._labels = torch.tensor(
            df.pop(label_name).values,
            device=self._device,
            dtype=torch.long,
        )
            
        self._features = self._converter.convert(df)
        
    def __len__(self) -> int:
        return len(self._features)

    def __getitem__(self, i: int) -> Tuple[torch.tensor, torch.tensor]:
        return self._features[i], self._labels[i]
