"""WCAG contrast for the site's palettes, and the smallest shift of a colour that reaches a target ratio."""
import colorsys, sys

def lum(h):
    h = h.lstrip('#'); r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    f = lambda c: c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4
    return 0.2126 * f(r) + 0.7152 * f(g) + 0.0722 * f(b)

def ratio(a, b):
    la, lb = sorted((lum(a), lum(b)), reverse=True); return (la + 0.05) / (lb + 0.05)

def shift(fg, bg, target):
    """Move fg's lightness away from bg's (keeping hue and saturation) until the ratio reaches target."""
    h = fg.lstrip('#'); r, g, b = (int(h[i:i+2], 16) / 255 for i in (0, 2, 4))
    hh, l, s = colorsys.rgb_to_hls(r, g, b); darker = lum(bg) > 0.18
    for _ in range(400):
        c = '#%02x%02x%02x' % tuple(round(v * 255) for v in colorsys.hls_to_rgb(hh, l, s))
        if ratio(c, bg) >= target: return c
        l = max(0, l - .005) if darker else min(1, l + .005)
    return c

if __name__ == "__main__":
    latte = dict(base='#eff1f5', mantle='#e6e9ef', text='#4c4f69', subtext1='#5c5f77', subtext0='#6c6f85', teal='#179299', blue='#1e66f5', mauve='#8839ef', green='#40a02b', pink='#ea76cb', sapphire='#209fb5', lavender='#7287fd')
    mocha = dict(base='#1e1e2e', mantle='#181825', surface0='#313244', text='#cdd6f4', subtext1='#bac2de', subtext0='#a6adc8', overlay2='#9399b2', teal='#94e2d5', blue='#89b4fa', mauve='#cba6f7', green='#a6e3a1', pink='#f5c2e7', sapphire='#74c7ec', lavender='#b4befe')
    for name, p in (("latte", latte), ("mocha", mocha)):
        print(name)
        for k, v in p.items():
            if k in ('base', 'mantle', 'surface0'): continue
            print(f"  {k:9} {v} on base {ratio(v, p['base']):5.2f}")
