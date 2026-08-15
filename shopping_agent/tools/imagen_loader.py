import os
import vertexai
from google.cloud import aiplatform
from vertexai.preview.vision_models import ImageGenerationModel

def load_imagen_edit_model(
    project_id: str = "adsp-s26-reccys",
    location: str = "us-central1",
    model_id: str = "imagen-3.0-capability-001",
):
    os.environ["GOOGLE_CLOUD_PROJECT"] = project_id
    os.environ["GOOGLE_PROJECT_ID"] = project_id
    os.environ["PROJECT_ID"] = project_id
    os.environ["GOOGLE_CLOUD_LOCATION"] = location
    os.environ["GOOGLE_CLOUD_REGION"] = location

    aiplatform.init(project=project_id, location=location)
    vertexai.init(project=project_id, location=location)

    print(f"[imagen_loader] project={project_id}, location={location}, model={model_id}")
    return ImageGenerationModel.from_pretrained(model_id)
