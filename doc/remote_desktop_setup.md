# Secure Remote Desktop: Tailscale + RustDesk

> **Goal**  Reach an Ubuntu VM that sits behind an unmanaged router/firewall from your macOS laptop, using **Tailscale** for the secure network overlay and **RustDesk** for a full‑GUI desktop session.
>
> • No inbound ports opened  • Identity‑aware access controls  • Session recording (optional)

---

## Prerequisites

| Component                        | Version / Notes                                           |
| -------------------------------- | --------------------------------------------------------- |
| **Ubuntu VM (Host)**                    | 20.04 LTS or newer (64‑bit)                               |
| **Tailscale account**            | Personal or Org; SSO/MFA recommended                      |
| **RustDesk**                     | Latest stable release (GUI app for client; host on Ubuntu) |
| **SSH access to VM (local/LAN)** | Needed for the initial install and restarting VM                   |

> 🛡️ If you are in a business/compliance environment, enable SSO + MFA in **Settings → Identity** inside the Tailscale admin console before proceeding.

---

## Step 0:  Create an Auth Key & Tag

1. In the Tailscale admin console, open **Settings → Keys**.
2. Click **Generate auth key**.
   - Type: `Reusable`
   - **Tags:** `tag:ubuntu-vm`
   - Optional: enable **Pre‑Approved** so the VM joins automatically.
3. **Copy** the key; you will paste it on the VM.

---

## Step 1:  Install & Log In – Client

### Tailscale

```bash
#-- macOS
  # Install via App Store or Homebrew
  brew install --cask tailscale   # ← easiest
#-- Linux
  # Install via website
  curl -fsSL https://tailscale.com/install.sh | sh

# Start the service
sudo tailscaled install-system-daemon

# Log in – browser popup will appear
sudo tailscale up

# Verify
tailscale status
```
Or download directly from [tailscale](https://tailscale.com/download)

### RustDesk

```bash
#-- macOS
brew install --cask rustdesk
#-- Linux
brew install --cask rustdesk
```
Or download directly from [RustDesk](https://rustdesk.com)

---

## Step 2:  Install – Ubuntu VM (Host)

### Tailscale

SSH (locally) or use the Proxmox console:

```bash
# 1  Install tailscale
curl -fsSL https://tailscale.com/install.sh | sh

# 2  Bring the VM into your tailnet
sudo tailscale up \
  --authkey tskey-REPLACE-ME \
  --hostname ubuntu-proxmox-01 \
  --advertise-tags=tag:ubuntu-vm

# 3  Enable Tailscale SSH (no static keys!)
sudo tailscale set --ssh

# 4  Check status & IP
tailscale ip -4       # → 100.x.y.z
sudo tailscale status
```

> **MagicDNS** (optional): In admin console **DNS → MagicDNS → Enable**. The VM is now reachable at `ubuntu-proxmox-01.tailnet.ts.net`.


### RustDesk

RustDesk provides both **host** (server) and **client** binaries. You only need the **host** component on the VM.

```bash
# Import RustDesk GPG + repo (Ubuntu 20.04/22.04)
wget -qO - https://apt.rustdesk.com/pub.gpg | sudo gpg --dearmor -o /usr/share/keyrings/rustdesk.gpg
echo "deb [arch=$(dpkg --print-architecture) signed-by=/usr/share/keyrings/rustdesk.gpg] https://apt.rustdesk.com/ $(lsb_release -cs) main" | sudo tee /etc/apt/sources.list.d/rustdesk.list

sudo apt update
sudo apt install rustdesk-server # (package name may be rustdesk-host on some releases)
```

Start the service and **bind only to the Tailscale interface** (extra safety):

```bash
sudo systemctl edit --full rustdesk.service
# Add or change:
# ExecStart=/usr/bin/rustdesk-host --host 100.x.y.z

sudo systemctl enable --now rustdesk.service
```

> Replace `100.x.y.z` with the VM’s Tailscale IP so RustDesk listens only on the private overlay network.

---

## Step 4  Connect via RustDesk over Tailscale

1. On your Mac, open **RustDesk**.
2. Enter the **Tailscale IP** (or MagicDNS name) of the VM.
3. Supply the connection password (set in RustDesk host) or use the RustDesk ID/relay if preferred.
4. You now have a full desktop session transported entirely inside the encrypted Tailscale mesh.

> Because traffic stays on the 100.\* overlay, no relays or public listeners are required – **zero inbound ports opened**.

---

## Step 5  (Optional) Harden & Audit

| Item                           | Command / Location                                                                         |
| ------------------------------ | ------------------------------------------------------------------------------------------ |
| **Least‑privilege ACL**        | Admin → *Access Controls* → JSON: allow only `user:you@company.com` to `tag:ubuntu-vm:*`.  |
| **Session recording**          | In the ACL `ssh` block, add `"record": true`.                                              |
| **Firewall**                   | UFW: allow inbound on `tailscale0` only.`sudo ufw allow in on tailscale0`                  |
| **Automatic updates**          | `sudo apt install unattended-upgrades`                                                     |
| **RustDesk service isolation** | Run under dedicated user; restrict with `SystemCallFilter` / `PrivateTmp` in systemd unit. |

---

## Appendix  Useful Commands

```bash
# Show Tailnet peers & latency
sudo tailscale status

# Diagnose path quality
sudo tailscale netcheck

# List SSH session recordings (if enabled)
#   in the Tailscale admin console → Machines → "Recordings"

# Upgrade Tailscale agent (Ubuntu)
sudo apt update && sudo apt install --only-upgrade tailscale
```

---

