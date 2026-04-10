# Hybrid Researcher - Remote Access via Cloudflare Tunnel

Password-protected launcher + Cloudflare quick tunnels for remote access to the Hybrid Researcher.

## Quick Start

```bash
# 1. Set password
export LAUNCHER_PASSWORD="your-secure-password"

# 2. Start quick tunnels (generates temporary public URLs)
./login/start-quick-tunnels.sh

# 3. Start the launcher (in a separate terminal)
./login/start-launcher.sh

# 4. Open the launcher URL printed by step 2
#    Log in -> Start App -> Open Hybrid Researcher App
```

## Port Assignments

| Service | Port | Purpose |
|---------|------|---------|
| Launcher | 8522 | Password-gated control panel |
| Main App | 8511 | Hybrid Researcher Streamlit app |

## How Quick Tunnels Work

- `start-quick-tunnels.sh` creates two temporary Cloudflare tunnels
- Each gets a random `*.trycloudflare.com` URL (changes on every restart)
- URLs are saved to `/tmp/hybrid-launcher-url.txt` and `/tmp/hybrid-app-url.txt`
- The launcher reads these files to display clickable links

## Stopping

```bash
# Stop tunnels (only this project, safe for brain-nw1)
pkill -f 'cloudflared tunnel --url http://localhost:8522'
pkill -f 'cloudflared tunnel --url http://localhost:8511'

# Stop launcher
lsof -ti:8522 | xargs -r kill -9

# Stop main app (also available via launcher UI)
lsof -ti:8511 | xargs -r kill -9
```

## Coexistence with brain-nw1

The quick tunnel scripts use **targeted** `pkill` by port URL, so they will not kill the `brain-nw1` tunnel or any other `cloudflared` processes. Both projects can run tunnels simultaneously.

## Log Files

| File | Content |
|------|---------|
| `/tmp/hybrid-launcher-tunnel.log` | Cloudflare tunnel log (launcher) |
| `/tmp/hybrid-app-tunnel.log` | Cloudflare tunnel log (app) |
| `/tmp/hybrid_researcher_app.log` | Streamlit app output |

## Upgrading to Permanent URLs

Quick tunnel URLs are temporary. For permanent URLs, you need a Cloudflare-managed domain:

```bash
# 1. Register/transfer a domain to Cloudflare (~$10/year for .com)
# 2. Create a named tunnel
cloudflared tunnel create hybrid-researcher

# 3. Set up DNS routes
cloudflared tunnel route dns hybrid-researcher hybrid-launcher.yourdomain.com
cloudflared tunnel route dns hybrid-researcher hybrid-app.yourdomain.com

# 4. Edit login/cloudflared-config.yml with your tunnel ID and domain
# 5. Copy to ~/.cloudflared/config.yml
# 6. Run the persistent tunnel
cloudflared tunnel run hybrid-researcher
```

## Troubleshooting

- **Tunnels fail to start**: Check if `cloudflared` is installed (`cloudflared version`)
- **URLs not extracted**: Wait longer (increase `sleep` in script) or check tunnel logs
- **Port already in use**: The scripts offer to kill the existing process
- **Config conflict**: The script temporarily moves `~/.cloudflared/config.yml` to force quick tunnel mode
