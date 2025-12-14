from huggingface_hub import HfApi
import os

api = HfApi(token=os.getenv("HF_TOURISM_PACKAGE_TOKEN"))
api.upload_folder(
    folder_path="tourism_project/deployment",     # the local folder containing your files
    repo_id="/laxmikantdeshpande/tourism-package-prediction",          # the target repo
    repo_type="space",                      # dataset, model, or space
    path_in_repo="",                          # optional: subfolder path inside the repo
)
