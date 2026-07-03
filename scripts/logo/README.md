# Logo Generator

This folder contains a browser-based SVG.js logo generator that creates a transparent 4×3 rounded table SVG.

## Usage

Open [logo_generator.html](logo_generator.html) in a browser from a served workspace to preview and download the result.

From the repository root, start a simple local server and then open the page in your browser:

```bash
python -m http.server 8000 --bind 127.0.0.1
```

Then visit [http://127.0.0.1:8000/scripts/logo/logo_generator.html](http://127.0.0.1:8000/scripts/logo/logo_generator.html).

The HTML file owns the page markup and the SVG.js CDN import. The stroke color is controlled by the `strokeColor` constant in [logo_generator.js](logo_generator.js). Edit that value and refresh the page to regenerate the SVG.

## Output

The generated SVG uses a transparent fill and the stroke color defined in code.
