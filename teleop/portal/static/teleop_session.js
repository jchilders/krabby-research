(function () {
  var PATH = typeof window.TELEOP_WS_PATH === 'string' ? window.TELEOP_WS_PATH : '/ws';
  var videosEl = document.getElementById('videos');
  var status = document.getElementById('status');
  var latencyEl = document.getElementById('latency');
  var streamStatus = document.getElementById('streamStatus');
  var debugEl = document.getElementById('debugRtc');
  var catalogListEl = document.getElementById('catalogList');
  var catalogInputEl = document.getElementById('catalogIds');
  var params = new URLSearchParams(location.search);
  var token = params.get('token');
  var qs = token ? '?token=' + encodeURIComponent(token) : '';
  var httpProto = location.protocol === 'https:' ? 'https:' : 'http:';
  var cfgUrl = httpProto + '//' + location.host + '/api/teleop-config' + qs;
  var wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  var wsUrl = wsProto + '//' + location.host + PATH + qs;

  var stunTurnServers = [{ urls: 'stun:stun.l.google.com:19302' }];
  var ws = null;
  var pc = null;
  var answerWaiter = null;
  var availableCatalogIds = [];
  var lastOffsetMs = null;
  var captureByCatalogIdNs = {};

  function waitGatheringComplete(p) {
    return new Promise(function (resolve) {
      if (p.iceGatheringState === 'complete') return resolve();
      p.addEventListener('icegatheringstatechange', function onGathering() {
        if (p.iceGatheringState === 'complete') {
          p.removeEventListener('icegatheringstatechange', onGathering);
          resolve();
        }
      });
    });
  }

  function updateDebug() {
    if (!debugEl || !pc) return;
    debugEl.textContent =
      'RTCPeerConnection: ' +
      pc.connectionState +
      ' | media: ' +
      pc.iceConnectionState +
      ' | signaling: ' +
      pc.signalingState;
  }

  function selectedCatalogIdsFromCheckboxes() {
    if (!catalogListEl) return [];
    var nodes = catalogListEl.querySelectorAll('input[type="checkbox"][data-catalog-id]:checked');
    var out = [];
    nodes.forEach(function (n) {
      var v = n.getAttribute('data-catalog-id');
      if (v) out.push(v);
    });
    return out;
  }

  function syncCatalogInputFromSelection(ids) {
    if (!catalogInputEl) return;
    catalogInputEl.value = ids.join(', ');
  }

  function renderCatalogList(ids) {
    if (!catalogListEl) return;
    if (!ids || !ids.length) {
      catalogListEl.textContent = 'Available sensors: waiting for robot hello...';
      return;
    }
    catalogListEl.innerHTML = '';
    var label = document.createElement('div');
    label.textContent = 'Available sensors (select order top-to-bottom):';
    catalogListEl.appendChild(label);
    ids.forEach(function (cid, idx) {
      var row = document.createElement('label');
      row.style.display = 'block';
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.setAttribute('data-catalog-id', cid);
      if (idx === 0) cb.checked = true;
      cb.addEventListener('change', function () {
        syncCatalogInputFromSelection(selectedCatalogIdsFromCheckboxes());
      });
      row.appendChild(cb);
      row.appendChild(document.createTextNode(' ' + cid));
      catalogListEl.appendChild(row);
    });
    syncCatalogInputFromSelection(selectedCatalogIdsFromCheckboxes());
  }

  function readNumStreams() {
    var el = document.querySelector('input[name="teleop_streams"]:checked');
    if (!el) return 1;
    var n = parseInt(el.value, 10);
    return n > 0 ? n : 1;
  }

  function selectedCatalogIdsInOrder() {
    return readCatalogIdsArray();
  }

  /** HAL RGB-D catalog ids in order; empty field -> ``[]`` (robot uses bootstrap / primary-only list). */
  function readCatalogIdsArray() {
    var selected = selectedCatalogIdsFromCheckboxes();
    if (selected.length > 0) {
      return selected;
    }
    if (!catalogInputEl) return [];
    return String(catalogInputEl.value || '')
      .split(',')
      .map(function (s) {
        return s.trim();
      })
      .filter(Boolean);
  }

  function helloPayload() {
    return {
      type: 'hello',
      role: 'browser',
      version: 1,
      catalog_ids: readCatalogIdsArray(),
    };
  }

  function offerPayload(sdp) {
    return { type: 'offer', sdp: sdp, catalog_ids: readCatalogIdsArray() };
  }

  async function startRtc(numStreams) {
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      status.textContent = 'WebSocket not ready';
      return;
    }
    if (pc) {
      pc.close();
      pc = null;
    }
    videosEl.innerHTML = '';
    if (streamStatus) {
      streamStatus.textContent = 'Requested ' + numStreams + ' stream(s); negotiating...';
    }
    status.textContent = 'Negotiating WebRTC...';

    pc = new RTCPeerConnection({ iceServers: stunTurnServers });
    pc.addEventListener('connectionstatechange', updateDebug);
    pc.addEventListener('iceconnectionstatechange', updateDebug);
    pc.addEventListener('signalingstatechange', updateDebug);
    updateDebug();

    var nTracks = 0;
    pc.ontrack = function (ev) {
      nTracks += 1;
      if (streamStatus) {
        streamStatus.textContent = 'Receiving ' + nTracks + ' / ' + numStreams + ' video track(s)';
      }
      var v = document.createElement('video');
      v.autoplay = true;
      v.playsInline = true;
      v.muted = true;
      v.style.maxWidth = '100%';
      v.style.background = '#111';
      v.srcObject = ev.streams[0];
      videosEl.appendChild(v);
    };

    var i;
    for (i = 0; i < numStreams; i += 1) {
      pc.addTransceiver('video', { direction: 'recvonly' });
    }
    var offer = await pc.createOffer();
    await pc.setLocalDescription(offer);
    await waitGatheringComplete(pc);

    var ans = await new Promise(function (resolve, reject) {
      answerWaiter = { resolve: resolve, reject: reject };
      ws.send(JSON.stringify(offerPayload(pc.localDescription.sdp)));
    });
    await pc.setRemoteDescription({ type: 'answer', sdp: ans.sdp });
    status.textContent = 'Playing';
    updateDebug();
  }

  function openWebSocket() {
    ws = new WebSocket(wsUrl);
    ws.onerror = function () {
      status.textContent = 'WebSocket error';
    };

    ws.onmessage = function (e) {
      var msg = JSON.parse(e.data);
      if (msg.type === 'error' && answerWaiter) {
        answerWaiter.reject(new Error(msg.message || 'server error'));
        answerWaiter = null;
        return;
      }
      if (msg.type === 'answer' && answerWaiter) {
        answerWaiter.resolve(msg);
        answerWaiter = null;
        return;
      }
      if (msg.type === 'hello_ack') {
        if (Array.isArray(msg.available_catalog_ids)) {
          availableCatalogIds = msg.available_catalog_ids
            .map(function (s) {
              return String(s || '').trim();
            })
            .filter(Boolean);
          renderCatalogList(availableCatalogIds);
        }
      }
      if (msg.type === 'pong' && typeof msg.t === 'number' && latencyEl) {
        var nowPerf = performance.now();
        var rttMs = Math.round(nowPerf - msg.t);
        if (typeof msg.t_wall_ms === 'number' && typeof msg.server_ms === 'number') {
          var t0 = msg.t_wall_ms;
          var t1 = msg.server_ms;
          var t3 = Date.now();
          var offset = t1 - ((t0 + t3) / 2.0); // robot wall-clock minus browser wall-clock
          if (lastOffsetMs === null) {
            lastOffsetMs = offset;
          } else {
            // Smooth jitter with light EMA.
            lastOffsetMs = (0.85 * lastOffsetMs) + (0.15 * offset);
          }
        }
        if (msg.capture_timestamps_ns && typeof msg.capture_timestamps_ns === 'object') {
          captureByCatalogIdNs = msg.capture_timestamps_ns;
        }
        var g2gText = 'g2g: n/a';
        var selected = selectedCatalogIdsInOrder();
        if (selected.length > 0 && lastOffsetMs !== null) {
          var cid = selected[0];
          var capNs = captureByCatalogIdNs[cid];
          if (typeof capNs === 'number') {
            var capMsRobot = capNs / 1e6;
            var capMsBrowser = capMsRobot - lastOffsetMs;
            var estG2g = Date.now() - capMsBrowser;
            if (isFinite(estG2g)) {
              g2gText = 'g2g~' + Math.max(0, Math.round(estG2g)) + ' ms (capture->render est, stream ' + cid + ')';
            }
          }
        }
        latencyEl.textContent =
          'RTT~' + rttMs + ' ms; offset~' +
          (lastOffsetMs === null ? 'n/a' : Math.round(lastOffsetMs) + ' ms') +
          '; ' + g2gText;
      }
    };

    ws.onopen = function () {
      try {
        ws.send(JSON.stringify(helloPayload()));
      } catch (e) {}
      startRtc(readNumStreams()).catch(function (err) {
        status.textContent = 'WebRTC error: ' + (err && err.message ? err.message : String(err));
      });
      setInterval(function () {
        if (ws.readyState === WebSocket.OPEN) {
          ws.send(JSON.stringify({ type: 'ping', t: performance.now(), t_wall_ms: Date.now() }));
        }
      }, 2000);
    };
  }

  function boot() {
    status.textContent = 'Loading config...';
    fetch(cfgUrl)
      .then(function (r) {
        if (!r.ok) return Promise.reject(new Error('config ' + r.status));
        return r.json();
      })
      .then(function (j) {
        if (j && Array.isArray(j.iceServers) && j.iceServers.length) {
          stunTurnServers = j.iceServers;
        }
        status.textContent = 'Connecting...';
        openWebSocket();
      })
      .catch(function () {
        status.textContent = 'Connecting...';
        openWebSocket();
      });
  }

  var applyBtn = document.getElementById('applyStreams');
  if (applyBtn) {
    applyBtn.addEventListener('click', function () {
      startRtc(readNumStreams()).catch(function (err) {
        status.textContent = 'WebRTC error: ' + (err && err.message ? err.message : String(err));
      });
    });
  }

  boot();
})();
