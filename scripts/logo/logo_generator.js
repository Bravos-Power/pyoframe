const tableConfig = {
  rows: 4,
  columns: 3,
  width: 280,
  height: 260,
  marginBottomTop: 5,
  marginLeftRight: 40,
  cornerRadius: 5,
  strokeWidth: 5,
};

const strokeColor = '#DC2626';
const squareColor = '#FB7185';
const speedLineColor = '#DC2626';
const logoSeed = 19770303;
const slantAngleDegrees = -15;
const pixelSquareSize = 10;
const pixelSquareGap = 3;
const outerBoundaryStrokeWidth = 12;
const dividerStrokeWidth = 5;
const speedLineYBase = 20;
const minSquareOpacity = 0.3;
const maxSquareOpacity = 0.9;
const speedLineConfig = [
  { x: 24, yOffset: 48, length: 56, width: 8, drawAfterTable: true },
  { x: 10, yOffset: 72, length: 42, width: 8, drawAfterTable: false },
  { x: 18, yOffset: 96, length: 27, width: 8, drawAfterTable: false },
];
const noiseOpacityGamma = 2.2;

function createSeededRandom(seed) {
  let value = seed >>> 0;

  return function random() {
    value += 0x6d2b79f5;
    let result = value;

    result = Math.imul(result ^ (result >>> 15), result | 1);
    result ^= result + Math.imul(result ^ (result >>> 7), result | 61);

    return ((result ^ (result >>> 14)) >>> 0) / 4294967296;
  };
}

const random = createSeededRandom(logoSeed);

function getRandomShade() {
  return minSquareOpacity + random() * (maxSquareOpacity - minSquareOpacity);
}

function getNoiseOpacity(normalizedY) {
  const fadeStart = 0;
  const fadeAmount = Math.max(0, Math.min(1, (normalizedY - fadeStart) / (1 - fadeStart)));
  const perceptualFade = Math.pow(fadeAmount, noiseOpacityGamma);

  return getRandomShade() * perceptualFade;
}

function addGridNoise(tableGroup) {
  const tableWidth = tableConfig.width - tableConfig.marginLeftRight * 2;
  const tableHeight = tableConfig.height - tableConfig.marginBottomTop * 2;
  const pixelStep = pixelSquareSize + pixelSquareGap;
  const columns = Math.floor((tableWidth + pixelSquareGap) / pixelStep);
  const rows = Math.floor((tableHeight + pixelSquareGap) / pixelStep);
  const noiseWidth = columns * pixelSquareSize + (columns - 1) * pixelSquareGap;
  const noiseHeight = rows * pixelSquareSize + (rows - 1) * pixelSquareGap;
  const startX = tableConfig.marginLeftRight + (tableWidth - noiseWidth) / 2;
  const startY = tableConfig.marginBottomTop + (tableHeight - noiseHeight) / 2;

  for (let rowIndex = 0; rowIndex < rows; rowIndex += 1) {
    const normalizedY = rows === 1 ? 1 : rowIndex / (rows - 1);

    for (let columnIndex = 0; columnIndex < columns; columnIndex += 1) {
      const x = startX + columnIndex * pixelStep;
      const y = startY + rowIndex * pixelStep;

      tableGroup
        .rect(pixelSquareSize, pixelSquareSize)
        .move(x, y)
        .fill({ color: squareColor, opacity: getNoiseOpacity(normalizedY) })
        .stroke('none');
    }
  }
}

function addSpeedLines(draw, shouldDrawAfterTable) {
  const speedLines = draw.group();

  speedLineConfig.forEach(({ x, yOffset, length, width, drawAfterTable }) => {
    if (drawAfterTable !== shouldDrawAfterTable) {
      return;
    }

    const y = speedLineYBase + yOffset;

    speedLines.line(x, y, x + length, y).stroke({
      color: speedLineColor,
      width,
      linecap: 'round',
    });
  });

  // Return the group so callers can explicitly arrange stacking with SVG.js helpers.
  return speedLines;
}

function createTableSvg(strokeColor) {
  const draw = globalThis.SVG().size(tableConfig.width, tableConfig.height);

  const tableWidth = tableConfig.width - tableConfig.marginLeftRight * 2;
  const tableHeight = tableConfig.height - tableConfig.marginBottomTop * 2;
  const columnWidth = tableWidth / tableConfig.columns;
  const rowHeight = tableHeight / tableConfig.rows;
  const slantFactor = Math.tan((slantAngleDegrees * Math.PI) / 180);
  const centerY = tableConfig.marginBottomTop + tableHeight / 2;

  draw
    .rect(tableConfig.width, tableConfig.height)
    .move(0, 0)
    .fill('none')
    .stroke('none');

  const clip = draw.clip().add(
    draw
      .rect(tableWidth, tableHeight)
      .move(tableConfig.marginLeftRight, tableConfig.marginBottomTop)
      .radius(tableConfig.cornerRadius)
  );

  // Create the back speed-lines.
  const backLines = addSpeedLines(draw, false);

  const tableGroup = draw.group();
  // Place backLines immediately before the tableGroup so they sit behind it
  // but remain above the SVG background rect.
  if (backLines && typeof tableGroup.before === 'function') tableGroup.before(backLines);
  tableGroup.attr('transform', `matrix(1 0 ${slantFactor} 1 ${-slantFactor * centerY} 0)`);
  tableGroup.clipWith(clip);

  // Put an opaque table base to occlude any back elements (like speed-lines).
  // Noise will be drawn on top of this base so it remains visible.
  tableGroup
    .rect(tableWidth, tableHeight)
    .move(tableConfig.marginLeftRight, tableConfig.marginBottomTop)
    .radius(tableConfig.cornerRadius)
    .fill('none')
    .stroke('none');

  // Draw noise inside a dedicated group on top of the base.
  const noiseGroup = tableGroup.group();
  addGridNoise(noiseGroup);

  // Draw inner dividers (above noise).
  for (let columnIndex = 1; columnIndex < tableConfig.columns; columnIndex += 1) {
    const x = tableConfig.marginLeftRight + columnIndex * columnWidth;
    tableGroup.line(x, tableConfig.marginBottomTop, x, tableConfig.marginBottomTop + tableHeight).stroke({
      color: strokeColor,
      width: dividerStrokeWidth,
      linecap: 'round',
    });
  }

  for (let rowIndex = 1; rowIndex < tableConfig.rows; rowIndex += 1) {
    const y = tableConfig.marginBottomTop + rowIndex * rowHeight;
    tableGroup.line(tableConfig.marginLeftRight, y, tableConfig.marginLeftRight + tableWidth, y).stroke({
      color: strokeColor,
      width: dividerStrokeWidth,
      linecap: 'round',
    });
  }

  // Draw outer boundary last so it sits above noise and dividers, then bring it to front.
  const borderRect = tableGroup
    .rect(tableWidth, tableHeight)
    .move(tableConfig.marginLeftRight, tableConfig.marginBottomTop)
    .radius(tableConfig.cornerRadius)
    .fill('none')
    .stroke({
      color: strokeColor,
      width: outerBoundaryStrokeWidth,
      linejoin: 'round',
    });
  if (borderRect && typeof borderRect.front === 'function') borderRect.front();

  // Create front speed-lines and ensure they sit above everything.
  const frontLines = addSpeedLines(draw, true);
  if (frontLines && typeof frontLines.front === 'function') frontLines.front();

  return draw.svg();
}

if (typeof document !== 'undefined') {
  const previewElement = document.getElementById('preview');
  const outputElement = document.getElementById('svg-output');
  const downloadLinkElement = document.getElementById('download-link');

  if (previewElement && outputElement && downloadLinkElement) {
    const svgText = createTableSvg(strokeColor);

    previewElement.innerHTML = svgText;
    outputElement.value = svgText;
    downloadLinkElement.href = `data:image/svg+xml;charset=utf-8,${encodeURIComponent(svgText)}`;
  }
}

export { createTableSvg };