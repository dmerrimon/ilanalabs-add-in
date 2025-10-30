# 🚀 Deploy Ilana Frontend to Azure Static Web Apps

## Prerequisites

1. **Azure Account**: Ensure you have an active Azure subscription
2. **GitHub Account**: Repository must be on GitHub for automatic deployment
3. **Azure CLI**: Install Azure CLI for command-line deployment (optional)

## Deployment Options

### Option 1: Azure Portal (Recommended)

#### Step 1: Create Azure Static Web App

1. **Login to Azure Portal**: https://portal.azure.com
2. **Create Resource** → Search "Static Web Apps" → Create
3. **Configure Basic Settings**:
   - **Subscription**: Your Azure subscription
   - **Resource Group**: Create new "ilana-frontend-rg"
   - **Name**: `ilana-frontend` (or your preferred name)
   - **Plan Type**: Free (sufficient for development)
   - **Azure Functions**: Leave as "East US 2" or nearest region

#### Step 2: Configure GitHub Integration

1. **Deployment Details**:
   - **Source**: GitHub
   - **GitHub Account**: Authorize Azure to access your GitHub
   - **Organization**: Your GitHub username
   - **Repository**: Select your Ilana repository
   - **Branch**: `main` (or `master`)

2. **Build Configuration**:
   - **Build Presets**: Custom
   - **App location**: `/ilana-frontend`
   - **Api location**: (leave empty)
   - **Output location**: (leave empty)

#### Step 3: Complete Deployment

1. Click **Review + Create** → **Create**
2. Azure will automatically:
   - Create GitHub workflow file
   - Deploy your frontend
   - Provide a live URL

### Option 2: Azure CLI Deployment

```bash
# Login to Azure
az login

# Create resource group
az group create --name ilana-frontend-rg --location eastus2

# Create static web app
az staticwebapp create \
    --name ilana-frontend \
    --resource-group ilana-frontend-rg \
    --source https://github.com/YOUR_USERNAME/ilana \
    --location eastus2 \
    --branch main \
    --app-location "/ilana-frontend" \
    --login-with-github
```

### Option 3: Manual File Upload

1. **Build Preparation**: Your files are already ready in `/ilana-frontend/`
2. **Azure Portal**: Create Static Web App as above
3. **Upload Files**: Use Azure CLI or portal to upload files directly

## Required Configuration Updates

### 1. Update Manifest.xml URLs

After deployment, update `/ilana-frontend/manifest.xml` with your actual Azure URL:

```xml
<!-- Replace YOUR_AZURE_STATIC_APP_URL with actual URL -->
<IconUrl DefaultValue="https://YOUR_APP_NAME.azurestaticapps.net/icon-32.png" />
<HighResolutionIconUrl DefaultValue="https://YOUR_APP_NAME.azurestaticapps.net/icon-80.png" />
<SourceLocation DefaultValue="https://YOUR_APP_NAME.azurestaticapps.net" />
```

### 2. Create Required Icons

Create these icon files in `/ilana-frontend/`:
- `icon-16.png` (16x16 pixels)
- `icon-32.png` (32x32 pixels) 
- `icon-80.png` (80x80 pixels)

### 3. Update Backend URL (if needed)

In `/ilana-frontend/ilana.js`, update the backend URL:
```javascript
const backendUrl = 'https://YOUR_BACKEND_URL.onrender.com';
```

## Expected Results

After successful deployment:

1. **Live URL**: `https://your-app-name.azurestaticapps.net`
2. **Auto SSL**: HTTPS enabled automatically
3. **Global CDN**: Fast worldwide access
4. **Custom Domain**: Can be configured if needed

## Testing Deployment

1. **Access URL**: Visit your Azure Static Web App URL
2. **Test Interface**: Verify Ilana interface loads correctly
3. **Test Word Add-in**: Install manifest in Word Online
4. **API Connection**: Test protocol analysis functionality

## Troubleshooting

### Common Issues:

1. **404 Errors**: Check `staticwebapp.config.json` routing
2. **CORS Issues**: Verify backend CORS configuration
3. **Icon Loading**: Ensure icon files are uploaded
4. **Manifest Errors**: Validate XML syntax and URLs

### Debug Steps:

1. **Check Logs**: Azure Portal → Static Web App → Functions/Overview
2. **Network Tab**: Browser developer tools for API calls  
3. **Console Errors**: Check browser console for JavaScript errors

## GitHub Integration

The deployment includes automatic CI/CD:
- **Push to main**: Triggers automatic deployment
- **Pull Requests**: Creates preview deployments
- **Branch Protection**: Configure as needed

## Custom Domain (Optional)

1. **Azure Portal**: Static Web App → Custom domains
2. **Add Domain**: Enter your custom domain
3. **DNS Configuration**: Add CNAME record to Azure URL
4. **SSL Certificate**: Automatically provisioned

## Monitoring

- **Azure Monitor**: Built-in monitoring and analytics
- **Application Insights**: Can be enabled for detailed telemetry
- **GitHub Actions**: View deployment history and logs

---

## 🎯 Quick Start Commands

```bash
# Clone and navigate
cd /Users/donmerriman/Ilana/ilana-frontend

# Verify files are ready
ls -la

# Deploy via Azure CLI (optional)
az staticwebapp create --name ilana-frontend --resource-group ilana-rg --source .
```

Your Ilana Protocol Intelligence frontend will be live and ready for Word add-in integration! 🚀