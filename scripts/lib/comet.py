import comet

def load_model(model_name):
    model_path = comet.download_model(model_name)
    model = comet.load_from_checkpoint(model_path).eval()
    return model