<script setup>
import { ref, computed, nextTick, onMounted, onBeforeUnmount } from 'vue'
import Papa from 'papaparse'
import L from 'leaflet'
import 'leaflet/dist/leaflet.css'
import csvFile from './ships.csv?raw'

// Fix default Leaflet icon paths in Vite / Webpack bundlers
delete L.Icon.Default.prototype._getIconUrl
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png'
})

/* ---------------------------------------------------------
   DATA
--------------------------------------------------------- */

const ships = ref([])
const selectedMmsi = ref('ALL')

const loading = ref(true)
const error = ref(null)

/* ---------------------------------------------------------
   CANVAS & MAP
--------------------------------------------------------- */

const mapElement = ref(null)
const canvas = ref(null)
const container = ref(null)

let map = null
let ctx = null
let animationFrame = null

const canvasWidth = ref(1000)
const canvasHeight = ref(600)

/* ---------------------------------------------------------
   SIMULATION
--------------------------------------------------------- */

const currentTime = ref(0)
const playing = ref(false)
const playbackDuration = ref(30)

let lastFrameTime = null

const playbackDurations = [120, 60, 30, 10, 5]

/* ---------------------------------------------------------
   TIME RANGE
--------------------------------------------------------- */

const startTime = ref(0)
const endTime = ref(0)

const currentDate = computed(() => {
  if (!currentTime.value) return ''
  return new Date(currentTime.value).toLocaleString()
})

function formatPlaybackDuration(seconds) {
  if (seconds >= 60) {
    return `${seconds / 60} min`
  }
  return `${seconds} sec`
}

/* ---------------------------------------------------------
   SELECTED SHIP
--------------------------------------------------------- */

const selectedShip = computed(() => {
  if (selectedMmsi.value === 'ALL') return null
  return ships.value.find(ship => ship.mmsi === selectedMmsi.value)
})

/* ---------------------------------------------------------
   DATA BOUNDS
--------------------------------------------------------- */

const bounds = ref({
  minLon: 0,
  maxLon: 1,
  minLat: 0,
  maxLat: 1
})

/* ---------------------------------------------------------
   LOAD CSV
--------------------------------------------------------- */

function loadCSV() {
  Papa.parse(csvFile, {
    header: true,
    skipEmptyLines: true,

    async complete(results) {
      try {
        processData(results.data)
        loading.value = false
        await nextTick()

        if (map) {
          map.invalidateSize()
        }
        resizeCanvas()
        fitMapToData()
        draw()
      } catch (err) {
        console.error(err)
        error.value = 'Failed to process AIS data.'
        loading.value = false
      }
    },

    error(err) {
      console.error(err)
      error.value = 'Failed to load CSV file.'
      loading.value = false
    }
  })
}

/* ---------------------------------------------------------
   PROCESS CSV
--------------------------------------------------------- */

function processData(rows) {
  const shipMap = new Map()

  for (const row of rows) {
    const latitude = Number(row.latitude)
    const longitude = Number(row.longitude)
    const sog = Number(row.sog)
    const cog = Number(row.cog)
    const timestamp = new Date(row.timestamp).getTime()

    if (
      !Number.isFinite(latitude) ||
      !Number.isFinite(longitude) ||
      !Number.isFinite(timestamp)
    ) {
      continue
    }

    const mmsi = String(row.mmsi)

    if (!shipMap.has(mmsi)) {
      shipMap.set(mmsi, {
        mmsi,
        imo: row.imo,
        vessel_name: row.vessel_name,
        callsign: row.callsign,
        vessel_type: row.vessel_type,
        draught_m: row.draught_m,
        destination: row.destination,
        points: []
      })
    }

    shipMap.get(mmsi).points.push({
      timestamp,
      latitude,
      longitude,
      sog: Number.isFinite(sog) ? sog : 0,
      cog: Number.isFinite(cog) ? cog : 0,
      heading: row.heading,
      rot: row.rot,
      nav_status: row.nav_status,
      anomaly_type: row.anomaly_type,
      spill_risk_level: row.spill_risk_level,
      event_description: row.event_description
    })
  }

  const processedShips = Array.from(shipMap.values())

  for (const ship of processedShips) {
    ship.points.sort((a, b) => a.timestamp - b.timestamp)
  }

  ships.value = processedShips
  calculateBounds()

  if (ships.value.length > 0) {
    startTime.value = Math.min(
      ...ships.value.flatMap(ship => ship.points.map(p => p.timestamp))
    )

    endTime.value = Math.max(
      ...ships.value.flatMap(ship => ship.points.map(p => p.timestamp))
    )

    currentTime.value = startTime.value
  }
}

/* ---------------------------------------------------------
   CALCULATE LONGITUDE / LATITUDE BOUNDS
--------------------------------------------------------- */

function calculateBounds() {
  const points = ships.value.flatMap(ship => ship.points)
  if (!points.length) return

  const longitudes = points.map(p => p.longitude)
  const latitudes = points.map(p => p.latitude)

  let minLon = Math.min(...longitudes)
  let maxLon = Math.max(...longitudes)
  let minLat = Math.min(...latitudes)
  let maxLat = Math.max(...latitudes)

  const lonPadding = Math.max((maxLon - minLon) * 0.05, 0.01)
  const latPadding = Math.max((maxLat - minLat) * 0.05, 0.01)

  bounds.value = {
    minLon: minLon - lonPadding,
    maxLon: maxLon + lonPadding,
    minLat: minLat - latPadding,
    maxLat: maxLat + latPadding
  }
}

/* ---------------------------------------------------------
   LONGITUDE/LATITUDE → CANVAS COORDINATES
--------------------------------------------------------- */

function toCanvasCoordinates(longitude, latitude) {
  if (!map) return { x: 0, y: 0 }
  const point = map.latLngToContainerPoint([latitude, longitude])
  return { x: point.x, y: point.y }
}

/* ---------------------------------------------------------
   RESIZE CANVAS
--------------------------------------------------------- */

function resizeCanvas() {
  if (!container.value || !canvas.value) return

  const rect = container.value.getBoundingClientRect()
  const dpr = window.devicePixelRatio || 1

  canvasWidth.value = rect.width
  canvasHeight.value = window.innerWidth <= 900
    ? Math.min(650, Math.max(300, rect.width * 0.55))
    : Math.max(300, rect.height)

  canvas.value.width = canvasWidth.value * dpr
  canvas.value.height = canvasHeight.value * dpr

  canvas.value.style.width = `${canvasWidth.value}px`
  canvas.value.style.height = `${canvasHeight.value}px`

  ctx = canvas.value.getContext('2d')
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0)

  if (map) map.invalidateSize()
  draw()
}

/* ---------------------------------------------------------
   REAL-WORLD MAP
--------------------------------------------------------- */

function initializeMap() {
  if (!mapElement.value) return

  map = L.map(mapElement.value, {
    zoomControl: true,
    preferCanvas: true
  })

  L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
    maxZoom: 19,
    attribution: '&copy; OpenStreetMap contributors'
  }).addTo(map)

  map.setView([35, 30], 4)
  map.on('move zoom resize', draw)
}

function fitMapToData() {
  if (!map || !ships.value.length) return

  map.fitBounds([
    [bounds.value.minLat, bounds.value.minLon],
    [bounds.value.maxLat, bounds.value.maxLon]
  ], {
    padding: [24, 24],
    maxZoom: 8
  })
}

/* ---------------------------------------------------------
   GET POSITION OF SHIP AT CURRENT TIME
--------------------------------------------------------- */

function getPositionAtTime(ship, time) {
  const points = ship.points
  if (!points.length) return null

  let latest = null
  for (const point of points) {
    if (point.timestamp <= time) {
      latest = point
    } else {
      break
    }
  }
  return latest
}

function drawShipIcon(x, y, color, cog, selected) {
  const scale = selected ? 1.15 : 0.9
  const rotation = (cog * Math.PI) / 180

  ctx.save()
  ctx.translate(x, y)
  ctx.rotate(rotation)
  ctx.scale(scale, scale)

  ctx.beginPath()
  ctx.moveTo(0, -10)
  ctx.lineTo(5, 5)
  ctx.lineTo(3, 9)
  ctx.lineTo(0, 6)
  ctx.lineTo(-3, 9)
  ctx.lineTo(-5, 5)
  ctx.closePath()

  ctx.fillStyle = color
  ctx.fill()

  ctx.strokeStyle = '#ffffff'
  ctx.lineWidth = 2
  ctx.stroke()

  ctx.restore()
}

/* ---------------------------------------------------------
   DRAW SHIP TRACK
--------------------------------------------------------- */

function drawShip(ship, index) {
  const selected = selectedMmsi.value === ship.mmsi
  const isAll = selectedMmsi.value === 'ALL'
  const hue = (index * 137.5) % 360
  const color = `hsl(${hue}, 70%, 45%)`

  const current = getPositionAtTime(ship, currentTime.value)
  if (!current) return

  const trail = ship.points.filter(point => point.timestamp <= current.timestamp)
  const trailCoordinates = trail.map(point => {
    const { x, y } = toCanvasCoordinates(point.longitude, point.latitude)
    return [x, y]
  })

  ctx.beginPath()
  trailCoordinates.forEach(([x, y], pointIndex) => {
    if (pointIndex === 0) {
      ctx.moveTo(x, y)
    } else {
      ctx.lineTo(x, y)
    }
  })

  ctx.strokeStyle = selected || isAll ? color : '#d1d5db'
  ctx.lineWidth = selected ? 3 : 1.5
  ctx.lineCap = 'round'
  ctx.lineJoin = 'round'
  ctx.globalAlpha = selectedMmsi.value === 'ALL' ? 1 : selected ? 1 : 0.3
  ctx.stroke()
  ctx.globalAlpha = 1

  const { x, y } = toCanvasCoordinates(current.longitude, current.latitude)

  drawShipIcon(x, y, color, Number(current.cog) || 0, selected)

  // Ship Label
  if (selected || selectedMmsi.value === 'ALL') {
    const name = ship.vessel_name || ship.mmsi
    ctx.fillStyle = '#111827'
    ctx.font = '12px Arial'
    ctx.textAlign = 'left'
    ctx.fillText(name, x + 10, y - 8)
  }
}

/* ---------------------------------------------------------
   DRAW EVERYTHING
--------------------------------------------------------- */

function draw() {
  if (!ctx) return

  ctx.clearRect(0, 0, canvasWidth.value, canvasHeight.value)

  const visibleShips = selectedMmsi.value === 'ALL'
    ? ships.value
    : ships.value.filter(ship => ship.mmsi === selectedMmsi.value)

  visibleShips.forEach((ship, index) => {
    drawShip(ship, index)
  })
}

/* ---------------------------------------------------------
   ANIMATION
--------------------------------------------------------- */

function animate(timestamp) {
  if (!playing.value) {
    lastFrameTime = null
    return
  }

  if (lastFrameTime === null) {
    lastFrameTime = timestamp
  }

  const delta = timestamp - lastFrameTime
  lastFrameTime = timestamp

  const dataDuration = endTime.value - startTime.value
  const simulationMillisecondsPerRealMillisecond =
    dataDuration / (playbackDuration.value * 1000)

  currentTime.value += delta * simulationMillisecondsPerRealMillisecond

  if (currentTime.value >= endTime.value) {
    currentTime.value = endTime.value
    playing.value = false
  }

  draw()

  if (playing.value) {
    animationFrame = requestAnimationFrame(animate)
  }
}

/* ---------------------------------------------------------
   CONTROLS
--------------------------------------------------------- */

function play() {
  if (currentTime.value >= endTime.value) {
    currentTime.value = startTime.value
  }
  lastFrameTime = null
  playing.value = true
  animationFrame = requestAnimationFrame(animate)
}

function pause() {
  playing.value = false
  lastFrameTime = null
  if (animationFrame) {
    cancelAnimationFrame(animationFrame)
    animationFrame = null
  }
}

function reset() {
  pause()
  currentTime.value = startTime.value
  draw()
}

function changePlaybackDuration(value) {
  playbackDuration.value = value
}

/* ---------------------------------------------------------
   TIMELINE
--------------------------------------------------------- */

function updateTimeline(event) {
  currentTime.value = Number(event.target.value)
  draw()
}

/* ---------------------------------------------------------
   SHIP SELECTION
--------------------------------------------------------- */

function selectShip(mmsi) {
  selectedMmsi.value = mmsi
  draw()
}

/* ---------------------------------------------------------
   SELECTED SHIP CURRENT DATA
--------------------------------------------------------- */

const selectedPosition = computed(() => {
  if (!selectedShip.value) return null
  return getPositionAtTime(selectedShip.value, currentTime.value)
})

/* ---------------------------------------------------------
   LIFECYCLE
--------------------------------------------------------- */

onMounted(() => {
  initializeMap()
  resizeCanvas()
  loadCSV()

  window.addEventListener('resize', resizeCanvas)
})

onBeforeUnmount(() => {
  window.removeEventListener('resize', resizeCanvas)
  if (animationFrame) {
    cancelAnimationFrame(animationFrame)
  }
  if (map) {
    map.remove()
  }
})
</script>

<template>
  <div class="app">
    <!-- HEADER -->
    <header class="header">
      <div>
        <h1>AIS Ship Track Visualizer</h1>
        <p>Longitude / latitude based vessel tracking</p>
      </div>
      <div class="ship-count">
        {{ ships.length }} ships
      </div>
    </header>

    <!-- ERROR BANNER -->
    <div v-if="error" class="message error">
      {{ error }}
    </div>

    <!-- MAIN WORKSPACE -->
    <div class="layout">
      <!-- MAIN VIEW -->
      <main class="main">
        <div ref="container" class="canvas-container">
          <!-- Non-destructive loading overlay -->
          <div v-if="loading" class="overlay-loading">
            <div class="spinner"></div>
            <span>Loading AIS data...</span>
          </div>

          <div ref="mapElement" class="leaflet-map"></div>
          <canvas ref="canvas"></canvas>
        </div>

        <!-- CONTROLS -->
        <section class="controls">
          <div class="buttons">
            <button @click="play" :disabled="playing || loading">
              ▶ Play
            </button>
            <button @click="pause" :disabled="!playing">
              ⏸ Pause
            </button>
            <button @click="reset" :disabled="loading">
              ↻ Reset
            </button>
          </div>

          <div class="speed">
            <span>Playback duration:</span>
            <button
              v-for="value in playbackDurations"
              :key="value"
              @click="changePlaybackDuration(value)"
              :class="{ active: playbackDuration === value }"
            >
              {{ formatPlaybackDuration(value) }}
            </button>
          </div>
        </section>

        <!-- TIMELINE -->
        <section class="timeline">
          <div class="time-label">
            <span>{{ new Date(startTime).toLocaleString() }}</span>
            <strong>{{ currentDate }}</strong>
            <span>{{ new Date(endTime).toLocaleString() }}</span>
          </div>

          <input
            type="range"
            :min="startTime"
            :max="endTime"
            :value="currentTime"
            :disabled="loading || startTime === endTime"
            @input="updateTimeline"
          />
        </section>
      </main>

      <!-- SIDEBAR -->
      <aside class="sidebar">
        <section>
          <h2>Ships</h2>

          <button
            class="ship-option"
            :class="{ selected: selectedMmsi === 'ALL' }"
            @click="selectShip('ALL')"
          >
            All ships
          </button>

          <button
            v-for="ship in ships"
            :key="ship.mmsi"
            class="ship-option"
            :class="{ selected: selectedMmsi === ship.mmsi }"
            @click="selectShip(ship.mmsi)"
          >
            <span>{{ ship.vessel_name || 'Unknown vessel' }}</span>
            <small>{{ ship.mmsi }}</small>
          </button>
        </section>

        <!-- SHIP DETAILS -->
        <section v-if="selectedShip" class="details">
          <h2>Ship details</h2>

          <div class="detail">
            <span>Vessel</span>
            <strong>{{ selectedShip.vessel_name || '-' }}</strong>
          </div>

          <div class="detail">
            <span>MMSI</span>
            <strong>{{ selectedShip.mmsi }}</strong>
          </div>

          <div class="detail">
            <span>IMO</span>
            <strong>{{ selectedShip.imo || '-' }}</strong>
          </div>

          <div class="detail">
            <span>Call sign</span>
            <strong>{{ selectedShip.callsign || '-' }}</strong>
          </div>

          <div v-if="selectedPosition" class="position">
            <h3>Current position</h3>

            <div class="detail">
              <span>Longitude</span>
              <strong>{{ selectedPosition.longitude.toFixed(5) }}</strong>
            </div>

            <div class="detail">
              <span>Latitude</span>
              <strong>{{ selectedPosition.latitude.toFixed(5) }}</strong>
            </div>

            <div class="detail">
              <span>Speed</span>
              <strong>{{ selectedPosition.sog }} knots</strong>
            </div>

            <div class="detail">
              <span>Course</span>
              <strong>{{ selectedPosition.cog }}°</strong>
            </div>

            <div class="detail">
              <span>Status</span>
              <strong>{{ selectedPosition.nav_status || '-' }}</strong>
            </div>

            <div
              v-if="selectedPosition.anomaly_type || selectedPosition.event_description"
              class="anomaly"
            >
              <strong>⚠ Anomaly detected</strong>
              <span>
                {{ selectedPosition.anomaly_type || selectedPosition.event_description }}
              </span>
              <small>
                Risk: {{ selectedPosition.spill_risk_level || 'Unknown' }}
              </small>
            </div>
          </div>
        </section>
      </aside>
    </div>
  </div>
</template>

<style scoped>
* {
  box-sizing: border-box;
}

:global(body) {
  margin: 0;
  background: linear-gradient(180deg, #eef5ff 0%, #f8fafc 100%);
}

.app {
  min-height: 100vh;
  background: linear-gradient(180deg, #eff6ff 0%, #f8fafc 100%);
  color: #0f172a;
  font-family: Inter, "Segoe UI", Arial, Helvetica, sans-serif;
}

/* HEADER */
.header {
  height: 80px;
  background: rgba(255, 255, 255, 0.88);
  backdrop-filter: blur(8px);
  border-bottom: 1px solid rgba(148, 163, 184, 0.2);
  box-shadow: 0 10px 25px rgba(15, 23, 42, 0.04);
  display: flex;
  align-items: center;
  justify-content: space-between;
  padding: 0 24px;
}

.header h1 {
  margin: 0;
  font-size: clamp(1.1rem, 2vw, 1.6rem);
  letter-spacing: -0.03em;
  color: #0f172a;
}

.header p {
  margin: 5px 0 0;
  color: #64748b;
  font-size: 13px;
}

.ship-count {
  background: linear-gradient(135deg, #dbeafe 0%, #eff6ff 100%);
  border: 1px solid #bfdbfe;
  color: #1d4ed8;
  font-size: 13px;
  font-weight: 700;
  padding: 8px 12px;
  border-radius: 999px;
  box-shadow: 0 6px 16px rgba(59, 130, 246, 0.08);
}

/* LAYOUT */
.layout {
  display: grid;
  grid-template-columns: minmax(0, 1fr) 300px;
  gap: 18px;
  padding: 18px;
  height: calc(100vh - 80px);
}

/* MAIN */
.main {
  min-width: 0;
  min-height: 0;
  display: grid;
  grid-template-rows: minmax(0, 1fr) auto auto;
  gap: 12px;
}

.canvas-container {
  position: relative;
  background: linear-gradient(180deg, #f8fafc 0%, #eef2ff 100%);
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 16px;
  overflow: hidden;
  min-height: 0;
  width: 100%;
  height: 100%;
  box-shadow: 0 20px 38px rgba(15, 23, 42, 0.08);
}

.leaflet-map,
canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
}

.leaflet-map {
  z-index: 1;
}

canvas {
  display: block;
  pointer-events: none;
  z-index: 2;
}

.overlay-loading {
  position: absolute;
  inset: 0;
  background: rgba(248, 250, 252, 0.88);
  backdrop-filter: blur(2px);
  display: flex;
  flex-direction: column;
  align-items: center;
  justify-content: center;
  gap: 10px;
  z-index: 1000;
  color: #475569;
  font-size: 14px;
  font-weight: 600;
}

.spinner {
  width: 26px;
  height: 26px;
  border: 3px solid #dfe7f5;
  border-top-color: #2563eb;
  border-radius: 50%;
  animation: spin 0.8s linear infinite;
}

@keyframes spin {
  to {
    transform: rotate(360deg);
  }
}

/* CONTROLS */
.controls {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 14px;
  padding: 12px 14px;
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 18px;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
}

.buttons,
.speed {
  display: flex;
  align-items: center;
  gap: 8px;
  flex-wrap: wrap;
}

button {
  border: 1px solid #dbe3f0;
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
  color: #334155;
  border-radius: 10px;
  padding: 8px 12px;
  cursor: pointer;
  font-size: 13px;
  font-weight: 600;
  transition: transform 0.15s ease, box-shadow 0.15s ease, border-color 0.15s ease, background 0.15s ease;
  box-shadow: 0 1px 2px rgba(15, 23, 42, 0.05);
}

button:hover:not(:disabled) {
  transform: translateY(-1px);
  background: linear-gradient(180deg, #f8fbff 0%, #eff6ff 100%);
  border-color: #bfdbfe;
  box-shadow: 0 8px 18px rgba(37, 99, 235, 0.08);
}

button:disabled {
  opacity: 0.5;
  cursor: default;
}

.speed span {
  font-size: 12px;
  color: #64748b;
  font-weight: 600;
  margin-right: 4px;
}

.speed button.active {
  background: linear-gradient(135deg, #111827 0%, #1f2937 100%);
  color: white;
  border-color: #111827;
  box-shadow: 0 10px 18px rgba(17, 24, 39, 0.15);
}

/* TIMELINE */
.timeline {
  background: rgba(255, 255, 255, 0.9);
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 14px;
  padding: 14px 16px;
  box-shadow: 0 8px 20px rgba(15, 23, 42, 0.04);
}

.timeline input {
  width: 100%;
  accent-color: #2563eb;
}

.time-label {
  display: flex;
  justify-content: space-between;
  align-items: center;
  gap: 8px;
  font-size: 11px;
  color: #64748b;
  margin-bottom: 10px;
}

.time-label strong {
  color: #0f172a;
  font-size: 12px;
  font-weight: 700;
}

/* SIDEBAR */
.sidebar {
  background: rgba(255, 255, 255, 0.92);
  border: 1px solid rgba(148, 163, 184, 0.25);
  border-radius: 16px;
  padding: 16px;
  height: fit-content;
  max-height: calc(100vh - 116px);
  overflow-y: auto;
  box-shadow: 0 12px 28px rgba(15, 23, 42, 0.06);
}

.sidebar h2 {
  margin: 0 0 12px;
  font-size: 15px;
  color: #0f172a;
  letter-spacing: -0.02em;
}

/* SHIP LIST */
.ship-option {
  width: 100%;
  display: flex;
  flex-direction: column;
  align-items: flex-start;
  margin-bottom: 8px;
  text-align: left;
  border-radius: 10px;
  padding: 10px 12px;
  background: linear-gradient(180deg, #ffffff 0%, #f8fafc 100%);
}

.ship-option small {
  margin-top: 4px;
  color: #94a3b8;
  font-weight: 600;
}

.ship-option.selected {
  background: linear-gradient(135deg, #e0f2fe 0%, #eff6ff 100%);
  border-color: #93c5fd;
  box-shadow: inset 0 0 0 1px rgba(59, 130, 246, 0.2);
}

/* DETAILS */
.details {
  border-top: 1px solid #e2e8f0;
  margin-top: 18px;
  padding-top: 18px;
}

.detail {
  display: flex;
  justify-content: space-between;
  gap: 10px;
  padding: 8px 0;
  border-bottom: 1px solid #f1f5f9;
  font-size: 12px;
}

.detail span {
  color: #64748b;
}

.detail strong {
  text-align: right;
  color: #0f172a;
  font-weight: 700;
}

/* POSITION */
.position {
  margin-top: 18px;
  background: linear-gradient(180deg, #f8fafc 0%, #f1f5f9 100%);
  border: 1px solid #e2e8f0;
  border-radius: 12px;
  padding: 12px;
}

.position h3 {
  font-size: 13px;
  margin: 0 0 8px;
  color: #0f172a;
}

/* ANOMALY */
.anomaly {
  margin-top: 15px;
  padding: 12px 10px;
  border: 1px solid #fecaca;
  background: linear-gradient(180deg, #fff5f5 0%, #fef2f2 100%);
  border-radius: 10px;
  display: flex;
  flex-direction: column;
  gap: 5px;
  font-size: 12px;
}

.anomaly strong {
  color: #b91c1c;
}

.anomaly small {
  color: #7f1d1d;
}

/* MESSAGES */
.message {
  margin: 16px 24px;
  padding: 12px 16px;
  border-radius: 10px;
  font-size: 14px;
  box-shadow: 0 10px 20px rgba(239, 68, 68, 0.08);
}

.error {
  background: linear-gradient(180deg, #fff1f2 0%, #fee2e2 100%);
  border: 1px solid #fca5a5;
  color: #b91c1c;
}

/* MOBILE */
@media (max-width: 900px) {
  .layout {
    grid-template-columns: 1fr;
    height: auto;
  }

  .main {
    display: block;
  }

  .canvas-container {
    height: 400px;
  }

  .sidebar {
    max-height: none;
    margin-top: 16px;
  }

  .controls {
    flex-direction: column;
    align-items: flex-start;
  }
}
</style>