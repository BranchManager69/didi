# Didi API Autostart Guide

This guide explains how to set up the Didi API to automatically start whenever your Lambda Labs instance is restarted or your persistent storage (`/home/ubuntu/degenduel-gpu`) is mounted.

## Automatic Setup

1. Run the installation script with sudo:

```bash
cd /home/ubuntu/degenduel-gpu/didi
sudo ./install-service.sh
```

This will:
- Install the systemd service
- Enable it to start on boot
- Start it immediately
- Show the current status

## Auto-Restart Monitor Setup

The auto-restart monitor watches for changes to UI files and automatically restarts the API server when they change:

1. Run the monitor installation script with sudo:

```bash
cd /home/ubuntu/degenduel-gpu/didi
sudo ./install-monitor.sh
```

This will:
- Install inotify-tools if needed
- Install and start the monitor service
- Configure it to restart automatically

## Manual Setup

If you prefer to set up the service manually:

1. Copy the service file to systemd:

```bash
sudo cp /home/ubuntu/degenduel-gpu/didi/didi-api.service /etc/systemd/system/
```

2. Reload systemd and enable the service:

```bash
sudo systemctl daemon-reload
sudo systemctl enable didi-api.service
sudo systemctl start didi-api.service
```

## Managing the Service

### Check Status

```bash
sudo systemctl status didi-api.service
sudo systemctl status didi-api-monitor.service
```

### View Logs

```bash
# View systemd logs for the service
sudo journalctl -u didi-api.service

# View application logs
cat /home/ubuntu/degenduel-gpu/didi/logs/didi_api.log

# View auto-restart logs
cat /home/ubuntu/degenduel-gpu/didi/logs/auto-restart.log
```

### Stop the Service

```bash
sudo systemctl stop didi-api.service
sudo systemctl stop didi-api-monitor.service
```

### Disable Autostart

```bash
sudo systemctl disable didi-api.service
sudo systemctl disable didi-api-monitor.service
```

## Configuration

The service is configured to:

- Run as the ubuntu user
- Use the GH200-optimized ultra profile with Llama-3-70B
- Listen on port 8000 by default
- Restart automatically if it crashes
- Load properly after the network is available
- Use all the correct environment variables for model paths

## Accessing the API

Once the service is running, you can access:

- Web interface: `http://localhost:8000/` (through SSH tunnel)
- API endpoints: `http://localhost:8000/api/...` (through SSH tunnel)

## File Monitoring

The auto-restart monitor watches for changes in:
- `/home/ubuntu/degenduel-gpu/didi/public` (HTML, CSS, JS files)
- `/home/ubuntu/degenduel-gpu/didi/scripts` (Python files)
- `/home/ubuntu/degenduel-gpu/didi/model_profiles` (JSON profiles)

When any of these files change, it automatically restarts the API server.

## Troubleshooting

If the service fails to start:

1. Check systemd logs:
```bash
sudo journalctl -u didi-api.service -n 50
```

2. Check application logs:
```bash
tail -50 /home/ubuntu/degenduel-gpu/didi/logs/didi_api.log
```

3. Ensure correct permissions:
```bash
sudo chmod +x /home/ubuntu/degenduel-gpu/didi/run_api.sh
sudo chmod +x /home/ubuntu/degenduel-gpu/didi/auto-restart.sh
sudo chown -R ubuntu:ubuntu /home/ubuntu/degenduel-gpu/didi/logs
```

4. Make sure all required packages are installed:
```bash
cd /home/ubuntu/degenduel-gpu/didi
pip install flask flask-cors gunicorn
sudo apt-get install inotify-tools
```

## Security Note

The service is configured to listen on all interfaces (0.0.0.0) but your firewall should restrict access to only specific IPs. Make sure your UFW configuration is properly set up.