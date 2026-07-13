# ApexFlow phone prototype

The simplest free prototype is Streamlit Community Cloud. After deployment,
the dashboard runs on Streamlit's servers, so the phone does not need your
computer or PowerShell to be running.

## 1. Put the project on GitHub

1. Sign in to GitHub and create a new repository, such as `apexflow-prototype`.
2. Keep the repository private if you do not want the source code public.
3. Upload the project files from this folder. Do not upload `__pycache__`,
   `.env`, or `.streamlit/secrets.toml`.
4. Confirm `dashboard.py`, `requirements.txt`, and the `.streamlit` folder are
   in the repository root.

## 2. Deploy it for free

1. Open https://share.streamlit.io and sign in.
2. Connect the GitHub account that owns the repository.
3. Select **Create app**, then **Yup, I have an app**.
4. Select the repository and the `main` branch.
5. Set the entrypoint file to `dashboard.py`.
6. In Advanced settings, choose Python 3.12. No secret is required for the
   current free Yahoo data provider.
7. Select **Deploy** and wait for the `.streamlit.app` URL.

## 3. Add it to the phone Home Screen

### iPhone

1. Open the deployed URL in Safari.
2. Tap **Share**.
3. Tap **Add to Home Screen**, rename it ApexFlow, and tap **Add**.

### Android

1. Open the deployed URL in Chrome.
2. Open the three-dot menu.
3. Tap **Add to Home screen** or **Install app**.

## Prototype behavior

- Local computer: the chart can use the five-second localhost bridge.
- Cloud/phone: the chart updates through Streamlit every 15 seconds and does
  not depend on port 8502 or your computer.
- In the cloud, use the main interval selector above the chart. The chart-local
  1m/5m/15m buttons require the local bridge.
- Free Yahoo data can be delayed or temporarily unavailable.
- A free Community Cloud app may sleep after inactivity and wake when visited.

