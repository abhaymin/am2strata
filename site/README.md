# The am2Strata showcase page

A static page for https://am2strata.am2.ai, served by Cloudflare Pages straight from this folder (no build step:
the project's "build output directory" is `site`). `_headers` sets the security and cache headers.

Images are copies of `docs/media/` sized for the web (the infographics are SVG, so they stay sharp in the viewer), and the hero background was made with am2Strata's media engine. Every screenshot and infographic opens in a full-screen viewer (zoom, pan, keys); the 3D stack and tilts turn off under reduced motion.
The font is Outfit (SIL OFL 1.1, `fonts/OFL.txt`).

## Ads

Two quiet, labelled slots: one under the videos, one text line in the footer. No pop-ups, nothing in the hero, nothing
animated, and a fixed size so the page does not jump. With `<meta name="am2-ads-publisher" content="">` empty they show
"From am2Strata" house ads (Sponsors, the lofi mix, the channel). To show paid ads, sign up at EthicalAds
(https://www.ethicalads.io/publishers/: privacy-friendly, no tracking cookies, so no consent banner) and put the
publisher id in that meta tag; `_headers` already allows its two domains. If no ad is served within 4 s, the house ad stays.

## Themes and accessibility (WCAG 2.2 AAA)

Colours follow Catppuccin: Mocha for dark, Latte for light, with Auto (the system's) as the default and a nav button to
choose; the choice is kept on the device. Text colours are Catppuccin's hues adjusted to 7:1 contrast on every background
they sit on (`tools/site/contrast.py` derives them); the bright hues are kept for decoration. The infographics are drawn by
`tools/site/infographics.py` in both themes, each with a text version on the page, and the script refuses to write one with
text under 7:1. Also: a skip link, a `main` landmark, 3 px focus rings, 44 px targets, a pause for all motion (and the system's
reduced-motion setting), new-tab notices, line lengths under 80 characters, and a glossary of the abbreviations.

Checked in a browser on 2026-11-16 by walking every text element in both themes: 336 checked, none under the AAA ratio
(lowest normal text 7.24:1 in Mocha, 7.81:1 in Latte); no horizontal scroll at 375 px.
