class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.

    Core idea:
    ----------
    Monitor the validation loss after each epoch.
    If it does not improve by at least "delta" for "patience" consecutive epochs,
    stop training to prevent overfitting and wasted computation.

    Typical usage:
    --------------
    early_stopping = EarlyStopping(patience=10, delta=1e-4)

    for epoch in range(epochs):
        train(...)
        val_loss = validate(...)

        early_stopping(val_loss, model)

        if early_stopping.early_stop:
            print("Early stopping triggered.")
            break
    """

    def __init__(self, patience=10, delta=0.0, verbose=False):
        """
        Parameters
        ----------
        patience : int
            Number of consecutive epochs with no sufficient improvement
            after which training will be stopped.

        delta : float
            Minimum change in validation loss required to qualify as an improvement.
            Helps ignore small noisy fluctuations.

        verbose : bool
            If True, prints messages when validation loss improves.

        path : str
            File path for saving the best model weights.
        """

        self.patience = patience
        self.delta = delta
        self.verbose = verbose

        self.best_loss = None
        self.counter = 0
        self.early_stop = False

    def __call__(self, val_loss):
        """
        Call this method after every validation phase.

        Parameters
        ----------
        val_loss : float
            Current epoch validation loss.

        model : torch.nn.Module
            Model being trained. Used to save the best checkpoint.
        """

        if self.best_loss is None:
            # First epoch → always treat as best
            self.best_loss = val_loss

        elif val_loss < self.best_loss - self.delta:
            # Improvement detected → reset counter
            self.best_loss = val_loss
            self.counter = 0
        else:
            # No significant improvement → increase patience counter
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping: {self.counter}/{self.patience}", flush = True)

            if self.counter >= self.patience:
                self.early_stop = True
