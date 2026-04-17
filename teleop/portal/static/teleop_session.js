(function () {
  var PATH = typeof window.TELEOP_WS_PATH === 'string' ? window.TELEOP_WS_PATH : '/ws';
  var videosEl = document.getElementById('videos');
  var status = document.getElementById('status');
  var latencyEl = document.getElementById('latency');
  var streamStatus = document.getElementById('streamStatus');
  var debugEl = document.getElementById('debugRtc');
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

  function readNumStreams() {
    var el = document.querySelector('input[name="teleop_streams"]:checked');
    if (!el) return 1;
    var n = parseInt(el.value, 10);
    return n > 0 ? n : 1;
  }

  /** HAL RGB-D catalog ids in order; empty field -> ``[]`` (robot uses bootstrap / primary-only list). */
  function readCatalogIdsArray() {
    var el = document.getElementById('catalogIds');
    if (!el) {
      return [];
    }
    return String(el.value || '')
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
      if (msg.type === 'pong' && typeof msg.t === 'number' && latencyEl) {
        latencyEl.textContent =
          'Signaling RTT ~' +
          Math.round(performance.now() - msg.t) +
          ' ms (signaling round-trip only, not glass-to-glass).';
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
          ws.send(JSON.stringify({ type: 'ping', t: performance.now() }));
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
