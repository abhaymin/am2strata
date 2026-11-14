//! What runs inside the VM: the first-boot cloud-init seed, the provisioning script (a small desktop — TigerVNC's
//! Xvnc, openbox, a taskbar — plus apt, Flatpak with Flathub, AppImage support and compressed swap), and the app
//! catalog with its install / remove / launch commands. All of it is plain shell run as root over SSH, so the agent and
//! the operator can install anything Debian, Flathub, a .deb or an AppImage offers.

use serde_json::{json, Value};

/// Bumped when the provisioning script changes, so an older VM is brought up to date on its next start.
pub const PROVISION_VERSION: u32 = 4;

/// The guest user (passwordless sudo; SSH key only, no password at all).
pub const USER: &str = "am2";

/// Single-quote `s` for a POSIX shell.
pub fn shq(s: &str) -> String {
    format!("'{}'", s.replace('\'', r"'\''"))
}

/// The cloud-init NoCloud seed: the user with our SSH key, and our own host key (so the host checks it is talking to
/// this VM, never trusting a key on first use).
pub fn user_data(user_key_pub: &str, host_key: &str, host_key_pub: &str) -> String {
    let indent = |s: &str| s.trim_end().lines().map(|l| format!("      {l}")).collect::<Vec<_>>().join("\n");
    format!(r#"#cloud-config
users:
  - name: {USER}
    shell: /bin/bash
    sudo: "ALL=(ALL) NOPASSWD:ALL"
    lock_passwd: true
    ssh_authorized_keys: ["{key}"]
ssh_pwauth: false
disable_root: true
ssh_deletekeys: true
ssh_genkeytypes: []
ssh_keys:
  ed25519_private: |
{priv_}
  ed25519_public: "{pub_}"
"#, key = user_key_pub.trim(), priv_ = indent(host_key), pub_ = host_key_pub.trim())
}

pub fn meta_data(instance: &str) -> String {
    format!("instance-id: {instance}\nlocal-hostname: am2-sandbox\n")
}

/// The base system and desktop, run as root. Idempotent: run again after a version bump or a display size change.
pub fn provision_script(width: u16, height: u16, timezone: &str) -> String {
    let tz = if !timezone.is_empty() && timezone.chars().all(|c| c.is_ascii_alphanumeric() || "/_+-".contains(c)) { timezone } else { "UTC" };
    format!(r##"set -e
timedatectl set-timezone {tz} 2>/dev/null || ln -sf /usr/share/zoneinfo/{tz} /etc/localtime
export DEBIAN_FRONTEND=noninteractive
U={USER}
UID_=$(id -u $U)
echo "== packages"
apt-get update -q
apt-get install -y -q --no-install-recommends \
  tigervnc-standalone-server tigervnc-tools openbox tint2 xterm pcmanfm xdotool wmctrl x11-utils x11-xserver-utils xclip \
  dbus-user-session dbus-x11 at-spi2-core xdg-utils desktop-file-utils fonts-dejavu-core fonts-noto-core \
  libfuse2t64 fuse3 flatpak ca-certificates curl wget gnupg unzip xz-utils systemd-zram-generator yad dnsmasq-base
echo "== flathub"
flatpak remote-add --system --if-not-exists flathub https://dl.flathub.org/repo/flathub.flatpakrepo
echo "== compressed swap"
printf '[zram0]\nzram-size = min(ram, 4096)\ncompression-algorithm = zstd\n' > /etc/systemd/zram-generator.conf
systemctl daemon-reload; systemctl start systemd-zram-setup@zram0.service || true
echo "== desktop"
H=/home/$U
install -d -o $U -g $U $H/.config/systemd/user/default.target.wants $H/.config/openbox $H/Applications $H/Desktop $H/Downloads $H/.local/share/applications
cat > $H/.config/systemd/user/am2-xvnc.service <<EOF
[Unit]
Description=am2Strata sandbox display (Xvnc :1, localhost only)
[Service]
ExecStart=/usr/bin/Xvnc :1 -geometry {width}x{height} -depth 24 -localhost -SecurityTypes None -rfbport 5901 -AlwaysShared -AcceptSetDesktopSize=0
Restart=always
[Install]
WantedBy=default.target
EOF
cat > $H/.config/systemd/user/am2-session.service <<EOF
[Unit]
Description=am2Strata sandbox desktop session (openbox)
After=am2-xvnc.service
Requires=am2-xvnc.service
[Service]
Environment=DISPLAY=:1
ExecStartPre=/bin/sh -c 'for i in \$(seq 1 100); do [ -e /tmp/.X11-unix/X1 ] && exit 0; sleep 0.1; done; exit 1'
ExecStartPre=/bin/systemctl --user import-environment DISPLAY
ExecStart=/usr/bin/openbox-session
Restart=always
[Install]
WantedBy=default.target
EOF
cat > $H/.config/openbox/autostart <<'EOF'
xsetroot -solid '#1f2a33' &
tint2 &
EOF
ln -sf ../am2-xvnc.service $H/.config/systemd/user/default.target.wants/am2-xvnc.service
ln -sf ../am2-session.service $H/.config/systemd/user/default.target.wants/am2-session.service
echo 'export DISPLAY=:1' > /etc/profile.d/am2-display.sh
cat > /usr/local/bin/am2-run <<'EOF'
#!/bin/sh
# start an app on the sandbox desktop, detached, with the session's display and D-Bus
export DISPLAY=:1 XDG_RUNTIME_DIR=/run/user/$(id -u) DBUS_SESSION_BUS_ADDRESS=unix:path=/run/user/$(id -u)/bus
cd "$HOME"
exec setsid -f "$@" >/dev/null 2>&1 </dev/null
EOF
chmod 755 /usr/local/bin/am2-run
mkdir -p /etc/am2/ext
cat > /usr/local/bin/am2-browser-policy <<'AM2POLICY'
#!/bin/sh
# one forced-extension list per browser, from the markers in /etc/am2/ext (Chromium takes only one list)
CRX=https://clients2.google.com/service/update2/crx
AMO=https://addons.mozilla.org/firefox/downloads/latest
ids=""; ff=""
if [ -e /etc/am2/ext/bitwarden ]; then
  ids="$ids\"nngceckbapebfimnlniiiahkandclblb;$CRX\","
  ff="$ff\"{{446900e4-71c2-419f-a6a7-df9c091e268b}}\": {{\"installation_mode\": \"force_installed\", \"install_url\": \"$AMO/bitwarden-password-manager/latest.xpi\"}},"
fi
doh=""; ffdoh=""
if [ -e /etc/am2/ext/ublock ]; then
  doh=', "DnsOverHttpsMode": "off"'; ffdoh=', "DNSOverHTTPS": {{"Enabled": false, "Locked": true}}'
  ids="$ids\"ddkjiahejlhfcafbddmgiahcphecmpfh;$CRX\","
  ff="$ff\"uBlock0@raymondhill.net\": {{\"installation_mode\": \"force_installed\", \"install_url\": \"$AMO/ublock-origin/latest.xpi\"}},"
fi
mkdir -p /etc/chromium/policies/managed /usr/lib/firefox-esr/distribution
rm -f /etc/chromium/policies/managed/am2-bitwarden.json
printf '{{"ExtensionInstallForcelist": [%s]%s}}\n' "${{ids%,}}" "$doh" > /etc/chromium/policies/managed/am2-extensions.json
printf '{{"policies": {{"ExtensionSettings": {{%s}}%s}}}}\n' "${{ff%,}}" "$ffdoh" > /usr/lib/firefox-esr/distribution/policies.json
AM2POLICY
chmod 755 /usr/local/bin/am2-browser-policy
cat > /usr/local/bin/am2-ads <<'AM2ADS'
#!/bin/sh
# am2-ads block [--lists pihole,adguard] [allowed.domain ...] | am2-ads allow | am2-ads status
# Blocking is a DNS filter for every program in this VM, as Pi-hole and AdGuard Home do it: a private dnsmasq answers
# 0.0.0.0 / :: for the listed domains and their subdomains. Lists: pihole (Pi-hole's default list, StevenBlack's
# unified hosts) and adguard (the AdGuard DNS filter). Allowed domains (and their subdomains) are left out.
set -e
L=/etc/am2/adblock; mkdir -p $L
fetch() {{ # name url: once a week
  if [ ! -s "$L/$1.src" ] || [ -n "$(find "$L/$1.src" -mtime +7)" ]; then
    curl -fsSL --retry 3 "$2" -o "$L/$1.tmp" && mv "$L/$1.tmp" "$L/$1.src"
  fi
  [ -s "$L/$1.src" ]
}}
off() {{
  systemctl disable --now am2-dns.service >/dev/null 2>&1 || true
  ln -sfn ../run/systemd/resolve/stub-resolv.conf /etc/resolv.conf
  sed -i '/^# am2-ads begin$/,/^# am2-ads end$/d' /etc/hosts
}}
case "$1" in
  block)
    shift; lists=pihole
    if [ "$1" = "--lists" ]; then lists=$2; shift 2; fi
    printf '%s\n' "$@" | tr 'A-Z' 'a-z' | sed 's/^\*\.//; /^$/d' > $L/allow.user
    : > $L/domains; cp $L/allow.user $L/allow
    for l in $(echo "$lists" | tr ',+' '  '); do
      case "$l" in
        pihole) fetch pihole https://raw.githubusercontent.com/StevenBlack/hosts/master/hosts
          awk '$1 == "0.0.0.0" && $2 != "0.0.0.0" {{ print $2 }}' $L/pihole.src >> $L/domains;;
        adguard) fetch adguard https://adguardteam.github.io/AdGuardSDNSFilter/Filters/filter.txt
          sed -n 's/^||\([a-z0-9._-]*\)\^$/\1/p' $L/adguard.src >> $L/domains
          # the list's own exceptions stay reachable
          sed -n 's/^@@||\([a-z0-9._-]*\)\^.*$/\1/p' $L/adguard.src >> $L/allow;;
        *) echo "unknown list $l (pihole, adguard)" >&2; exit 2;;
      esac
    done
    sort -u $L/domains | awk 'NR==FNR {{ if ($0 != "") allow[$0]=1; next }}
      {{ d=$0; keep=1; n=split(d, p, ".");
        for (i=1; i<=n; i++) {{ s=p[i]; for (j=i+1; j<=n; j++) s=s "." p[j]; if (s in allow) {{ keep=0; break }} }}
        if (keep) print "address=/" d "/#" }}' $L/allow - > $L/block.conf
    rm -f $L/domains
    printf 'listen-address=127.0.0.2\nbind-interfaces\nno-hosts\nresolv-file=/run/systemd/resolve/resolv.conf\ncache-size=10000\nconf-file=%s/block.conf\n' $L > $L/dnsmasq.conf
    printf '[Unit]\nDescription=am2Strata ad and tracker DNS filter\nAfter=systemd-resolved.service\n[Service]\nExecStart=/usr/sbin/dnsmasq -k --conf-file=%s/dnsmasq.conf\nRestart=always\n[Install]\nWantedBy=multi-user.target\n' $L > /etc/systemd/system/am2-dns.service
    sed -i '/^# am2-ads begin$/,/^# am2-ads end$/d' /etc/hosts
    systemctl daemon-reload; systemctl enable am2-dns.service >/dev/null 2>&1; systemctl restart am2-dns.service
    for i in 1 2 3 4 5 6 7 8 9 10; do systemctl is-active -q am2-dns.service && break; sleep 0.5; done
    if ! systemctl is-active -q am2-dns.service; then off; echo "the DNS filter did not start: $(journalctl -u am2-dns -n 3 --no-pager -o cat | tr '\n' ' ')" >&2; exit 1; fi
    rm -f /etc/resolv.conf; printf 'nameserver 127.0.0.2\noptions edns0\n' > /etc/resolv.conf
    echo "$lists" > $L/lists
    touch /etc/am2/ext/ublock; am2-browser-policy
    echo "blocking $(wc -l < $L/block.conf) ad and tracker domains and their subdomains (lists: $lists); let through: $(tr '\n' ' ' < $L/allow.user)";;
  allow)
    off; rm -f /etc/am2/ext/ublock $L/allow.user $L/lists; am2-browser-policy; echo "ads allowed";;
  status)
    if systemctl is-active -q am2-dns.service && [ ! -L /etc/resolv.conf ]; then
      echo "blocked lists=$(cat $L/lists 2>/dev/null) allow=$(tr '\n' ',' < $L/allow.user 2>/dev/null | sed 's/,$//')"
    else echo allowed; fi;;
  *) echo "usage: am2-ads block [--lists pihole,adguard] [allowed.domain ...] | allow | status" >&2; exit 2;;
esac
AM2ADS
chmod 755 /usr/local/bin/am2-ads
cat > /usr/local/bin/am2-ads-app <<'AM2ADSAPP'
#!/bin/sh
# "Ads and trackers": a small window on this desktop to block or allow ads (it runs am2-ads)
set -f
T="Ads and trackers"
while :; do
  st=$(am2-ads status)
  on=FALSE; allow=""; lists=pihole
  case "$st" in blocked*)
    on=TRUE
    lists=$(echo "$st" | sed -n 's/.*lists=\([^ ]*\).*/\1/p' | tr ',' '+'); [ -n "$lists" ] || lists=pihole
    allow=$(echo "$st" | sed -n 's/.*allow=\(.*\)$/\1/p');;
  esac
  opts=""; for o in pihole adguard pihole+adguard; do [ "$o" = "$lists" ] && o="^$o"; opts="$opts!$o"; done
  out=$(yad --title="$T" --center --width=500 --form --separator='|' \
    --text="A DNS filter for every program on this desktop, plus uBlock in the browsers.\n<b>pihole</b>: Pi-hole's default list. <b>adguard</b>: the AdGuard DNS filter.\nList a site under <i>let through</i> when a job must see it as it is." \
    --field="Block ads and trackers:CHK" "$on" --field="List:CB" "${{opts#!}}" \
    --field="Let through (domains, comma separated)" "$allow" \
    --button="Close:1" --button="Apply:0") || exit 0
  block=$(echo "$out" | cut -d'|' -f1); lists=$(echo "$out" | cut -d'|' -f2); allow=$(echo "$out" | cut -d'|' -f3)
  if [ "$block" = TRUE ]; then msg=$(sudo am2-ads block --lists "$lists" $(echo "$allow" | tr ',' ' ') 2>&1); else msg=$(sudo am2-ads allow 2>&1); fi
  yad --title="$T" --center --width=460 --text="$msg\n\nRestart the browser for it to apply there." --button="OK:0" || true
done
AM2ADSAPP
chmod 755 /usr/local/bin/am2-ads-app
printf '[Desktop Entry]\nType=Application\nName=Ads and trackers\nComment=Block or allow ads on this desktop\nExec=am2-ads-app\nIcon=security-high\nCategories=Settings;\n' > /usr/share/applications/am2-ads.desktop
if [ -e /etc/chromium/policies/managed/am2-bitwarden.json ]; then touch /etc/am2/ext/bitwarden; fi
am2-browser-policy
chown -R $U:$U $H/.config
loginctl enable-linger $U
systemctl start user@$UID_.service
sleep 1
sudo -u $U XDG_RUNTIME_DIR=/run/user/$UID_ systemctl --user daemon-reload
sudo -u $U XDG_RUNTIME_DIR=/run/user/$UID_ systemctl --user restart am2-xvnc.service am2-session.service
echo "== tidy"
apt-get clean
fstrim -a >/dev/null 2>&1 || true
sync; echo 3 > /proc/sys/vm/drop_caches
echo "== ready"
"##)
}

/// How an app is installed.
#[derive(Clone, Copy, PartialEq, Debug)]
pub enum Source { Base, Apt, Flatpak, Script }

pub struct App {
    pub id: &'static str,
    pub name: &'static str,
    pub about: &'static str,
    pub source: Source,
    /// apt packages, a Flathub id, or a script
    pub what: &'static str,
    pub remove: &'static str,
    /// a shell test that succeeds when it is installed
    pub check: &'static str,
    /// the command `am2-run` starts (empty: nothing to open)
    pub launch: &'static str,
}

pub const CATALOG: &[App] = &[
    App { id: "terminal", name: "Terminal", about: "xterm, always there", source: Source::Base, what: "", remove: "",
          check: "command -v xterm", launch: "xterm" },
    App { id: "files", name: "Files", about: "PCManFM file manager", source: Source::Base, what: "", remove: "",
          check: "command -v pcmanfm", launch: "pcmanfm" },
    App { id: "ads", name: "Ads and trackers", about: "a window on the desktop to block or allow ads", source: Source::Base,
          what: "", remove: "", check: "command -v am2-ads-app", launch: "am2-ads-app" },
    App { id: "chromium", name: "Chromium", about: "the open-source browser behind Chrome (Debian)", source: Source::Apt,
          what: "chromium", remove: "chromium", check: "dpkg -s chromium >/dev/null 2>&1",
          launch: "chromium --no-first-run --password-store=basic" },
    App { id: "firefox", name: "Firefox ESR", about: "Mozilla's extended-support browser (Debian)", source: Source::Apt,
          what: "firefox-esr", remove: "firefox-esr", check: "dpkg -s firefox-esr >/dev/null 2>&1", launch: "firefox-esr" },
    App { id: "bitwarden", name: "Bitwarden", about: "the password manager's desktop app (Flathub); you sign in yourself",
          source: Source::Flatpak, what: "com.bitwarden.desktop", remove: "com.bitwarden.desktop",
          check: "flatpak info com.bitwarden.desktop >/dev/null 2>&1", launch: "flatpak run com.bitwarden.desktop" },
    App { id: "bitwarden-extension", name: "Bitwarden in the browsers",
          about: "installs the Bitwarden extension into Chromium and Firefox ESR by policy; you sign in yourself",
          source: Source::Script,
          what: "mkdir -p /etc/am2/ext && touch /etc/am2/ext/bitwarden && am2-browser-policy",
          remove: "rm -f /etc/am2/ext/bitwarden && am2-browser-policy",
          check: "test -e /etc/am2/ext/bitwarden", launch: "" },
    App { id: "vscode", name: "VS Code", about: "Microsoft's code editor (Flathub)", source: Source::Flatpak,
          what: "com.visualstudio.code", remove: "com.visualstudio.code",
          check: "flatpak info com.visualstudio.code >/dev/null 2>&1", launch: "flatpak run com.visualstudio.code" },
    App { id: "libreoffice", name: "LibreOffice", about: "Writer, Calc and Impress (Debian)", source: Source::Apt,
          what: "libreoffice-writer libreoffice-calc libreoffice-impress libreoffice-gtk3",
          remove: "libreoffice-writer libreoffice-calc libreoffice-impress libreoffice-gtk3 libreoffice-core libreoffice-common",
          check: "dpkg -s libreoffice-writer >/dev/null 2>&1", launch: "libreoffice" },
    App { id: "thunderbird", name: "Thunderbird", about: "mail (Debian)", source: Source::Apt, what: "thunderbird",
          remove: "thunderbird", check: "dpkg -s thunderbird >/dev/null 2>&1", launch: "thunderbird" },
    App { id: "keepassxc", name: "KeePassXC", about: "an offline password manager (Debian)", source: Source::Apt,
          what: "keepassxc", remove: "keepassxc", check: "dpkg -s keepassxc >/dev/null 2>&1", launch: "keepassxc" },
    App { id: "gimp", name: "GIMP", about: "image editor (Debian)", source: Source::Apt, what: "gimp", remove: "gimp",
          check: "dpkg -s gimp >/dev/null 2>&1", launch: "gimp" },
    App { id: "vlc", name: "VLC", about: "media player (Debian)", source: Source::Apt, what: "vlc", remove: "vlc",
          check: "dpkg -s vlc >/dev/null 2>&1", launch: "vlc" },
    App { id: "obsidian", name: "Obsidian", about: "notes (Flathub)", source: Source::Flatpak, what: "md.obsidian.Obsidian",
          remove: "md.obsidian.Obsidian", check: "flatpak info md.obsidian.Obsidian >/dev/null 2>&1",
          launch: "flatpak run md.obsidian.Obsidian" },
    App { id: "tailscale", name: "Tailscale", about: "joins the VM to your tailnet (Tailscale's own Debian repository); Open shows the sign-in link, you sign in yourself",
          source: Source::Script,
          what: "set -e; curl -fsSL https://pkgs.tailscale.com/stable/debian/trixie.noarmor.gpg -o /usr/share/keyrings/tailscale-archive-keyring.gpg; \
                 curl -fsSL https://pkgs.tailscale.com/stable/debian/trixie.tailscale-keyring.list -o /etc/apt/sources.list.d/tailscale.list; \
                 export DEBIAN_FRONTEND=noninteractive; apt-get update -q; apt-get install -y -q tailscale; systemctl enable --now tailscaled",
          remove: "tailscale logout 2>/dev/null || true; export DEBIAN_FRONTEND=noninteractive; apt-get remove -y -q tailscale; \
                   rm -f /etc/apt/sources.list.d/tailscale.list /usr/share/keyrings/tailscale-archive-keyring.gpg",
          check: "dpkg -s tailscale >/dev/null 2>&1",
          launch: "xterm -T Tailscale -hold -e sh -c 'sudo tailscale up; echo; tailscale status'" },
    App { id: "netbird", name: "NetBird", about: "joins the VM to your NetBird network (NetBird's own Debian repository); Open shows the sign-in link, you sign in yourself",
          source: Source::Script,
          what: "set -e; curl -fsSL https://pkgs.netbird.io/debian/public.key | gpg --dearmor --yes -o /usr/share/keyrings/netbird-archive-keyring.gpg; \
                 echo 'deb [signed-by=/usr/share/keyrings/netbird-archive-keyring.gpg] https://pkgs.netbird.io/debian stable main' > /etc/apt/sources.list.d/netbird.list; \
                 export DEBIAN_FRONTEND=noninteractive; apt-get update -q; apt-get install -y -q netbird; netbird service install 2>/dev/null || true; netbird service start 2>/dev/null || true",
          remove: "netbird down 2>/dev/null || true; netbird service stop 2>/dev/null || true; netbird service uninstall 2>/dev/null || true; \
                   export DEBIAN_FRONTEND=noninteractive; apt-get remove -y -q netbird; rm -f /etc/apt/sources.list.d/netbird.list /usr/share/keyrings/netbird-archive-keyring.gpg",
          check: "dpkg -s netbird >/dev/null 2>&1",
          launch: "xterm -T NetBird -hold -e sh -c 'sudo netbird up; echo; sudo netbird status'" },
    App { id: "mullvad", name: "Mullvad VPN", about: "Mullvad's app (Mullvad's own Debian repository); you enter your account number yourself",
          source: Source::Script,
          what: "set -e; curl -fsSL https://repository.mullvad.net/deb/mullvad-keyring.asc -o /usr/share/keyrings/mullvad-keyring.asc; \
                 echo \"deb [signed-by=/usr/share/keyrings/mullvad-keyring.asc arch=$(dpkg --print-architecture)] https://repository.mullvad.net/deb/stable stable main\" > /etc/apt/sources.list.d/mullvad.list; \
                 export DEBIAN_FRONTEND=noninteractive; apt-get update -q; apt-get install -y -q mullvad-vpn; \
                 sleep 2; mullvad lan set allow || true",
          remove: "export DEBIAN_FRONTEND=noninteractive; apt-get remove -y -q mullvad-vpn; rm -f /etc/apt/sources.list.d/mullvad.list /usr/share/keyrings/mullvad-keyring.asc",
          check: "dpkg -s mullvad-vpn >/dev/null 2>&1", launch: "/opt/Mullvad\\ VPN/mullvad-vpn --no-sandbox" },
    App { id: "surfshark", name: "Surfshark", about: "Surfshark's app (Flathub); you sign in yourself. If it cannot connect, use Surfshark's WireGuard or OpenVPN file with \"WireGuard and OpenVPN\"",
          source: Source::Flatpak, what: "com.surfshark.Surfshark", remove: "com.surfshark.Surfshark",
          check: "flatpak info com.surfshark.Surfshark >/dev/null 2>&1", launch: "flatpak run com.surfshark.Surfshark" },
    App { id: "vpn", name: "WireGuard and OpenVPN", about: "wg-quick and openvpn for any VPN provider's config file (put it in the VM, then: sudo wg-quick up ./x.conf, or sudo openvpn x.ovpn)",
          source: Source::Apt, what: "wireguard-tools openvpn", remove: "wireguard-tools openvpn",
          check: "dpkg -s wireguard-tools >/dev/null 2>&1 && dpkg -s openvpn >/dev/null 2>&1", launch: "" },
    App { id: "dev-tools", name: "Developer tools", about: "git, build-essential, Python (pip, venv), Node.js and npm",
          source: Source::Apt, what: "git build-essential python3-pip python3-venv nodejs npm",
          remove: "build-essential python3-pip python3-venv nodejs npm",
          check: "dpkg -s build-essential >/dev/null 2>&1 && dpkg -s nodejs >/dev/null 2>&1", launch: "" },
    App { id: "docker", name: "Docker", about: "containers inside the VM (docker.io); the am2 user may use it", source: Source::Apt,
          what: "docker.io", remove: "docker.io", check: "dpkg -s docker.io >/dev/null 2>&1", launch: "" },
];

/// The ad-blocking switch's command: `block` with the domains left out (for a site whose ads a job needs), or `allow`.
pub const AD_LISTS: [&str; 2] = ["pihole", "adguard"];

pub fn ads_script(block: bool, lists: &[String], allow: &[String]) -> Result<String, String> {
    if !block {
        return Ok("am2-ads allow".into());
    }
    let lists: Vec<&str> = if lists.is_empty() { vec!["pihole"] } else { lists.iter().map(String::as_str).collect() };
    if let Some(bad) = lists.iter().find(|l| !AD_LISTS.contains(l)) {
        return Err(format!("unknown list {bad:?}: pihole (Pi-hole's default list) or adguard (the AdGuard DNS filter)"));
    }
    let mut doms = vec![];
    for d in allow {
        let d = d.trim().trim_start_matches("*.").to_lowercase();
        let d = d.trim_start_matches("https://").trim_start_matches("http://").split('/').next().unwrap_or("").to_string();
        if d.is_empty() { continue; }
        if !d.contains('.') || !d.chars().all(|c| c.is_ascii_alphanumeric() || ".-".contains(c)) {
            return Err(format!("{d:?} is not a domain name"));
        }
        doms.push(d);
    }
    Ok(format!("am2-ads block --lists {} {}", lists.join(","), doms.join(" ")).trim_end().to_string())
}

pub fn app(id: &str) -> Option<&'static App> {
    CATALOG.iter().find(|a| a.id == id)
}

/// The root shell script that installs (or removes) a catalog app.
pub fn app_script(a: &App, install: bool) -> String {
    let apt = "export DEBIAN_FRONTEND=noninteractive; apt-get update -q && apt-get install -y -q";
    match (a.source, install) {
        (Source::Base, _) => "echo 'part of the base system'".into(),
        (Source::Apt, true) => {
            let extra = if a.id == "docker" { format!(" && usermod -aG docker {USER}") } else { String::new() };
            format!("{apt} {}{extra}", a.what)
        }
        (Source::Apt, false) => format!("export DEBIAN_FRONTEND=noninteractive; apt-get remove -y -q {} && apt-get autoremove -y -q", a.remove),
        (Source::Flatpak, true) => format!("flatpak install --system -y --noninteractive flathub {}", a.what),
        (Source::Flatpak, false) => format!("flatpak uninstall --system -y --noninteractive {} && flatpak uninstall --system -y --unused", a.remove),
        (Source::Script, true) => a.what.to_string(),
        (Source::Script, false) => a.remove.to_string(),
    }
}

/// Anything else: Debian packages, a Flathub app, a .deb from a URL, or an AppImage from a URL (it lands in
/// ~/Applications with a menu entry). Checked here so a request cannot smuggle shell syntax into a package name.
pub fn package_script(kind: &str, target: &str, install: bool) -> Result<String, String> {
    let safe_pkg = |p: &str| !p.is_empty() && p.chars().all(|c| c.is_ascii_alphanumeric() || "+-.:_".contains(c));
    let target = target.trim();
    match (kind, install) {
        ("apt", _) => {
            let pkgs: Vec<&str> = target.split_whitespace().collect();
            if pkgs.is_empty() || !pkgs.iter().all(|p| safe_pkg(p)) {
                return Err("apt: give Debian package names (letters, digits, + - . : _)".into());
            }
            Ok(if install { format!("export DEBIAN_FRONTEND=noninteractive; apt-get update -q && apt-get install -y -q {}", pkgs.join(" ")) }
               else { format!("export DEBIAN_FRONTEND=noninteractive; apt-get remove -y -q {} && apt-get autoremove -y -q", pkgs.join(" ")) })
        }
        ("flatpak", _) => {
            if !safe_pkg(target) || !target.contains('.') {
                return Err("flatpak: give a Flathub app id, e.g. org.gimp.GIMP".into());
            }
            Ok(if install { format!("flatpak install --system -y --noninteractive flathub {target}") }
               else { format!("flatpak uninstall --system -y --noninteractive {target}") })
        }
        ("deb", true) => {
            if !target.starts_with("https://") { return Err("deb: give an https:// URL of a .deb".into()); }
            Ok(format!("set -e; f=$(mktemp --suffix=.deb); curl -fsSL -o \"$f\" {}; export DEBIAN_FRONTEND=noninteractive; \
                        apt-get update -q; apt-get install -y -q \"$f\"; rm -f \"$f\"", shq(target)))
        }
        ("deb", false) => package_script("apt", target, false),
        ("appimage", true) => {
            if !target.starts_with("https://") { return Err("appimage: give an https:// URL of an .AppImage".into()); }
            let name: String = target.rsplit('/').next().unwrap_or("app").split(['?', '#']).next().unwrap_or("app")
                .chars().filter(|c| c.is_ascii_alphanumeric() || "._-".contains(*c)).collect();
            let name = if name.is_empty() { "app.AppImage".to_string() } else { name };
            let stem = name.trim_end_matches(".AppImage").trim_end_matches(".appimage").to_string();
            Ok(format!("set -e; H=/home/{USER}; install -d -o {USER} -g {USER} $H/Applications $H/.local/share/applications; \
                        curl -fsSL -o $H/Applications/{name} {url}; chmod 755 $H/Applications/{name}; chown {USER}:{USER} $H/Applications/{name}; \
                        printf '[Desktop Entry]\\nType=Application\\nName={stem}\\nExec=%s --no-sandbox\\n' $H/Applications/{name} > $H/.local/share/applications/{stem}.desktop; \
                        chown {USER}:{USER} $H/.local/share/applications/{stem}.desktop; echo \"installed $H/Applications/{name}\"",
                       url = shq(target)))
        }
        ("appimage", false) => {
            let name: String = target.rsplit('/').next().unwrap_or("").chars().filter(|c| c.is_ascii_alphanumeric() || "._-".contains(*c)).collect();
            if name.is_empty() { return Err("appimage: give the AppImage's file name".into()); }
            let stem = name.trim_end_matches(".AppImage").to_string();
            Ok(format!("rm -f /home/{USER}/Applications/{name} /home/{USER}/.local/share/applications/{stem}.desktop"))
        }
        (other, _) => Err(format!("unknown kind {other:?}: apt, flatpak, deb or appimage")),
    }
}

/// One shell line that prints, per catalog app, `id 0|1`.
pub fn catalog_check_script() -> String {
    CATALOG.iter().map(|a| format!("if {}; then echo '{} 1'; else echo '{} 0'; fi", a.check, a.id, a.id)).collect::<Vec<_>>().join("; ")
}

pub fn catalog_json(installed: &std::collections::HashMap<String, bool>) -> Value {
    json!(CATALOG.iter().map(|a| json!({
        "id": a.id, "name": a.name, "about": a.about, "launch": !a.launch.is_empty(),
        "source": match a.source { Source::Base => "base", Source::Apt => "apt", Source::Flatpak => "flatpak", Source::Script => "setup" },
        "installed": installed.get(a.id).copied(), "removable": a.source != Source::Base,
    })).collect::<Vec<_>>())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn quoting_and_package_names_are_safe() {
        assert_eq!(shq("it's"), r"'it'\''s'");
        assert!(package_script("apt", "gimp; rm -rf /", true).is_err());
        assert!(package_script("apt", "$(reboot)", true).is_err());
        assert!(package_script("flatpak", "org.gimp.GIMP && x", true).is_err());
        assert!(package_script("deb", "http://x/y.deb", true).is_err(), "https only");
        let s = package_script("apt", "git  htop", true).unwrap();
        assert!(s.ends_with("apt-get install -y -q git htop"), "{s}");
        let a = package_script("appimage", "https://example.com/dl/My App-1.2.AppImage?x=1", true).unwrap();
        assert!(a.contains("Applications/MyApp-1.2.AppImage") && a.contains("'https://example.com/dl/My App-1.2.AppImage?x=1'"), "{a}");
    }

    #[test]
    fn the_seed_carries_our_keys_and_no_password() {
        let u = user_data("ssh-ed25519 AAAA user", "-----BEGIN OPENSSH PRIVATE KEY-----\nabc\n-----END OPENSSH PRIVATE KEY-----\n", "ssh-ed25519 BBBB host");
        assert!(u.starts_with("#cloud-config\n"));
        assert!(u.contains("ssh_authorized_keys: [\"ssh-ed25519 AAAA user\"]"));
        assert!(u.contains("      abc\n") && u.contains("ed25519_public: \"ssh-ed25519 BBBB host\""));
        assert!(u.contains("lock_passwd: true") && u.contains("ssh_pwauth: false"));
    }

    #[test]
    fn every_catalog_app_has_install_remove_and_check() {
        for a in CATALOG {
            assert!(!a.check.is_empty(), "{}", a.id);
            if a.source != Source::Base {
                assert!(!app_script(a, true).is_empty() && !app_script(a, false).is_empty(), "{}", a.id);
            }
        }
        assert!(app("bitwarden").is_some() && app("chromium").is_some());
    }

    #[test]
    fn the_ad_switch_takes_only_domain_names() {
        assert_eq!(ads_script(false, &[], &[]).unwrap(), "am2-ads allow");
        assert_eq!(ads_script(true, &[], &[]).unwrap(), "am2-ads block --lists pihole");
        assert_eq!(ads_script(true, &["pihole".into(), "adguard".into()], &["https://News.Example.com/page".into(), "*.shop.io".into()]).unwrap(),
                   "am2-ads block --lists pihole,adguard news.example.com shop.io");
        assert!(ads_script(true, &["easylist; reboot".into()], &[]).is_err());
        assert!(ads_script(true, &[], &["x; reboot".into()]).is_err());
        assert!(ads_script(true, &[], &["localhost".into()]).is_err());
    }
}
