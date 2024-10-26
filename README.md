# NUSISS Project - Presidential Election ABSA Streamlit UI

This project is related to [this repo](https://github.com/destonedbob/nusiss-project-presidential-absa-system/tree/main). It is meant to the the streamlit UI to the other repo.

The main branch is for streamlit to read for the Streamlit Cloud App. To run locally on Windows, please use the `windows-branch` branch instead.

The app currently can be ran on streamlit only for visualization. The model prediction through Streamlit Cloud is not workable because streamlit crashes during model prediction, likely due to computational resource constraints.

If you wish to try out the UI, you may clone this repo's other branch (`windows-branch`) on windows, install the requirements and run streamlit locally.

```
python -m venv myVenv
"myVenv/Scripts/activate"
pip install -r requirements.txt
streamlit run main.py
```
