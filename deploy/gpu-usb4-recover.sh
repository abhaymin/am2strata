#!/usr/bin/env bash
# Brings up NVIDIA GPUs that failed at boot. A GPU on USB4 can lose its link for a few seconds just as the driver loads
# ("fallen off the bus"); nothing retries it when the link returns. This waits for the card to answer, removes it from
# the PCI bus, rescans, and creates the /dev/nvidia<N> files (nvidia-modprobe may not be installed). Run as root, at boot
# (deploy/gpu-usb4-recover.service) or by hand. A card that is working is never touched.
set -u
tries=${GPU_RECOVER_TRIES:-3}

all_cards() {   # every NVIDIA display device on the PCI bus
    for d in /sys/bus/pci/devices/*; do
        [ "$(cat "$d/vendor" 2>/dev/null)" = 0x10de ] || continue
        case "$(cat "$d/class" 2>/dev/null)" in 0x03*) basename "$d" ;; esac
    done
}
working() {     # the PCI addresses nvidia-smi can talk to, lower case
    nvidia-smi --query-gpu=pci.bus_id --format=csv,noheader 2>/dev/null | tr 'A-F' 'a-f' | sed 's/^0000\(....:\)/\1/'
}
make_nodes() {  # /dev/nvidia<minor> for every card the driver has
    for f in /proc/driver/nvidia/gpus/*/information; do
        m=$(awk -F: '/Device Minor/ {gsub(/[ \t]/, "", $2); print $2}' "$f" 2>/dev/null)
        [ -n "$m" ] && [ ! -e "/dev/nvidia$m" ] && mknod -m 666 "/dev/nvidia$m" c 195 "$m"
    done
}
missing() { local w; w=$(working); for c in $(all_cards); do grep -q "$c" <<<"$w" || echo "$c"; done; }

# the USB4 link settles in the first seconds after the driver loads
for _ in $(seq 1 45); do
    [ "$(awk '{print int($1)}' /proc/uptime)" -ge 25 ] && break
    sleep 1
done
make_nodes
for try in $(seq 1 "$tries"); do
    bad=$(missing)
    [ -z "$bad" ] && { echo "all NVIDIA GPUs are up: $(nvidia-smi -L | wc -l)"; exit 0; }
    for c in $bad; do
        s=${c#0000:}
        for _ in $(seq 1 30); do [ "$(setpci -s "$s" VENDOR_ID 2>/dev/null)" = 10de ] && break; sleep 1; done
        echo "restarting $c (try $try of $tries)"
        echo 1 > "/sys/bus/pci/devices/$c/remove"
    done
    sleep 5
    echo 1 > /sys/bus/pci/rescan
    sleep 10
    make_nodes
done
bad=$(missing)
[ -z "$bad" ] && { echo "all NVIDIA GPUs are up: $(nvidia-smi -L | wc -l)"; exit 0; }
echo "still not up: $bad" >&2
exit 1
