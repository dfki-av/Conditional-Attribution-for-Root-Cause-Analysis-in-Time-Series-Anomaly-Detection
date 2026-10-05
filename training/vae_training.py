"""Training entry point for the existing VAE fit implementation."""


def train_variational_autoencoder(model, training_data, validation_data=None, epochs=1000, verbose=0, device=None):
    """Train using the model's unchanged original fit loop."""
    model.fit_on_data(training_data, validation_data, epochs, verbose, device=device)
