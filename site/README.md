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
