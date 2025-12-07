# Deploying DCRM Streamlit App to DigitalOcean

This guide outlines how to deploy your DCRM Analyzer Pro application to DigitalOcean's App Platform.

## Prerequisites

1.  **DigitalOcean Account**: You need an account on [DigitalOcean](https://www.digitalocean.com/).
2.  **GitHub Repository**: Your code must be pushed to a GitHub repository (private or public).
3.  **Google API Key**: You will need your `GOOGLE_API_KEY` for the application to function.

## Step 1: Push Code to GitHub

Ensure your latest code (including the new `Dockerfile` and `.dockerignore`) is committed and pushed to your GitHub repository.

```bash
git add .
git commit -m "Add Docker deployment configuration"
git push origin main
```

## Step 2: Create App on DigitalOcean

1.  Log in to the **DigitalOcean Control Panel**.
2.  Click **Create** (green button at top right) -> **Apps**.
3.  **Choose Source**: Select **GitHub**.
4.  **Repository**: Select your repository (e.g., `final_DCRM_3PHASE_REVIEWED`).
5.  **Branch**: Select `main` (or your working branch).
6.  **Source Directory**: `/` (default).
7.  Click **Next**.

## Step 3: Configure Resources

1.  DigitalOcean should auto-detect the `Dockerfile` and select **Dockerfile** as the build strategy.
2.  **Service Name**: You can rename it (e.g., `dcrm-api`).
3.  **HTTP Port**: Ensure this is set to **5000** (Flask default).
4.  **Instance Size**: 
    *   Basic / Pro plans work well. 
    *   **Recommendation**: 1 GB RAM minimum (approx $6/mo) is good for the API.
5.  Click **Next**.

## Step 4: Environment Variables

1.  Click **Edit** next to your service (or go to extracting environment variables step).
2.  Add the following **Global Environment Variables**:
    *   **Key**: `GOOGLE_API_KEY`
    *   **Value**: *[Paste your actual Google API Key here]*
    *   **Encrypt**: Checked (Recommended)
3.  Click **Save**.
4.  Click **Next**.

## Step 5: Review & Deploy

1.  Review the details (Region, Plan, Cost).
2.  Click **Create Resources**.

DigitalOcean will now build your Docker image and deploy it. This process usually takes 3-5 minutes.

## Step 6: Access Your App

Once deployment is successful, you will see a **Live URL** (e.g., `https://dcrm-app-xyz.ondigitalocean.app/`). Click it to access your DCRM Analyzer!

## Troubleshooting

-   **Build Failed**: Check the "Activity" tab for build logs. Verify requirements installation.
-   **Health Check Failed**: Ensure port 8501 is exposed in Dockerfile (we handled this) and App Platform settings.
-   **App Crashes**: Check "Runtime Logs". Often due to missing API keys or memory limits. If OOM (Out of Memory), upgrade to a 1GB or 2GB instance.
