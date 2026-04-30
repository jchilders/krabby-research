(function () {
  var PATH = typeof window.TELEOP_WS_PATH === 'string' ? window.TELEOP_WS_PATH : '/ws';
  var videosEl = document.getElementById('videos');
  var status = document.getElementById('status');
  var latencyEl = document.getElementById('latency');
  var streamStatus = document.getElementById('streamStatus');
  var debugEl = document.getElementById('debugRtc');
  var mediaDebugEl = document.getElementById('teleopMediaDebug');
  var catalogListEl = document.getElementById('catalogList');
  var controllerStatusEl = document.getElementById('controllerStatus');
  var controlChannelStatusEl = document.getElementById('controlChannelStatus');
  var controlSendStatusEl = document.getElementById('controlSendStatus');
  var controllerStateMirrorEl = document.getElementById('controllerStateMirror');
  var params = new URLSearchParams(location.search);
  var token = params.get('token');
  var qs = token ? '?token=' + encodeURIComponent(token) : '';
  var httpProto = location.protocol === 'https:' ? 'https:' : 'http:';
  var cfgUrl = httpProto + '//' + location.host + '/api/teleop-config' + qs;
  var wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
  var wsUrl = wsProto + '//' + location.host + PATH + qs;

  /** Radius [0,1) for radial stick deadzone (resting jitter); raise if a pad needs more margin. */
  var TELEOP_GAMEPAD_STICK_DEADZONE = 0.1;

  var stunTurnServers = [{ urls: 'stun:stun.l.google.com:19302' }];
  var ws = null;
  var pc = null;
  var controlDc = null;
  var answerWaiter = null;
  var availableCatalogIds = [];
  var lastOffsetMs = null;
  var captureByCatalogIdNs = {};
  var gamepadLoopTimer = null;
  var controlSendCount = 0;
  var lastControlSendMs = 0;
  /** First offer must run after ``hello_ack`` so ``catalog_ids`` matches checkboxes (not ``[]``). */
  var initialRtcStarted = false;
  var initialRtcFallbackTimer = null;

  /** Captured when the last offer JSON is sent (for on-page diagnostics). */
  var lastOfferCatalogIdsSnapshot = [];
  var lastOfferRequestedRecvonlyLines = 0;
  var lastOfferSentAtIso = '';

  /** Filled in-order from ``ontrack`` for the active ``pc``. */
  var receivedVideoTracksMeta = [];

  var mediaDebugPollTimer = null;

  /** Per-stick radial deadzone: zero inside radius ``zone``, smooth rescale toward full ±1 beyond. */
  function stickPairAfterDeadzone(x, y, zone) {
    if (!(zone > 0)) {
      return [x, y];
    }
    var m = Math.sqrt(x * x + y * y);
    if (m <= zone) {
      return [0.0, 0.0];
    }
    var gain = (m - zone) / (m * (1.0 - zone));
    function clampAxis(v) {
      if (v < -1) return -1;
      if (v > 1) return 1;
      return v;
    }
    return [clampAxis(x * gain), clampAxis(y * gain)];
  }

  function readGamepadState() {
    var pads = (navigator.getGamepads && navigator.getGamepads()) || [];
    var p = null;
    for (var i = 0; i < pads.length; i += 1) {
      if (pads[i]) {
        p = pads[i];
        break;
      }
    }
    if (!p) {
      return {
        LT: false, LB: false, LS: false, RS: false, RT: false, RB: false,
        LX: 0.0, LY: 0.0, RX: 0.0, RY: 0.0
      };
    }
    function b(idx) { return !!(p.buttons && p.buttons[idx] && p.buttons[idx].pressed); }
    function a(idx) {
      var v = (p.axes && typeof p.axes[idx] === 'number') ? p.axes[idx] : 0.0;
      if (v < -1) return -1;
      if (v > 1) return 1;
      return v;
    }
    var lx0 = a(0);
    var ly0 = a(1);
    var rx0 = a(2);
    var ry0 = a(3);
    var left = stickPairAfterDeadzone(lx0, ly0, TELEOP_GAMEPAD_STICK_DEADZONE);
    var right = stickPairAfterDeadzone(rx0, ry0, TELEOP_GAMEPAD_STICK_DEADZONE);
    return {
      LT: b(6), LB: b(4), LS: b(10), RS: b(11), RT: b(7), RB: b(5),
      LX: left[0],
      LY: left[1],
      RX: right[0],
      RY: right[1]
    };
  }

  function firstConnectedGamepad() {
    var pads = (navigator.getGamepads && navigator.getGamepads()) || [];
    for (var i = 0; i < pads.length; i += 1) {
      if (pads[i]) return pads[i];
    }
    return null;
  }

  function updateControllerDebugPanel() {
    var gp = firstConnectedGamepad();
    if (controllerStatusEl) {
      if (gp) {
        controllerStatusEl.textContent =
          'Controller: connected (' + (gp.id || 'unknown') + ', index ' + gp.index + ')';
      } else {
        controllerStatusEl.textContent = 'Controller: not connected';
      }
    }
    if (controlChannelStatusEl) {
      var dcState = controlDc ? controlDc.readyState : 'n/a';
      controlChannelStatusEl.textContent =
        'Control channel (WebRTC krabby-control-v1): ' + dcState;
    }
    if (controlSendStatusEl) {
      if (controlDc && controlDc.readyState === 'open') {
        if (lastControlSendMs > 0) {
          controlSendStatusEl.textContent =
            'Relayed to robot: ' + controlSendCount + ' msgs, last ' +
            (Date.now() - lastControlSendMs) + 'ms ago';
        } else {
          controlSendStatusEl.textContent =
            'Relayed to robot: channel open (first control frame pending…)';
        }
      } else {
        controlSendStatusEl.textContent =
          'Relayed to robot: idle (no open WebRTC channel; local capture above still updates)';
      }
    }
    if (controllerStateMirrorEl) {
      controllerStateMirrorEl.textContent = JSON.stringify(readGamepadState(), null, 2);
    }
  }

  function stopGamepadLoop() {
    if (gamepadLoopTimer !== null) {
      clearInterval(gamepadLoopTimer);
      gamepadLoopTimer = null;
    }
  }

  function startGamepadLoop() {
    stopGamepadLoop();
    controlSendCount = 0;
    lastControlSendMs = 0;
    gamepadLoopTimer = setInterval(function () {
      if (!controlDc || controlDc.readyState !== 'open') return;
      var st = readGamepadState();
      controlDc.send(
        JSON.stringify({
          type: 'control',
          sent_browser_ms: Date.now(),
          state: st
        })
      );
      controlSendCount += 1;
      lastControlSendMs = Date.now();
    }, 20); // 50 Hz
  }

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

  function stopMediaDebugPoll() {
    if (mediaDebugPollTimer !== null) {
      clearInterval(mediaDebugPollTimer);
      mediaDebugPollTimer = null;
    }
  }

  /** Parse ``m-line`` indexes for bundled video recv in local offer SDP (SDP order ≈ browser transceiver creation order). */
  function sdpVideoRecvonlyMidOrder(sdpText) {
    if (!sdpText || typeof sdpText !== 'string') {
      return [];
    }
    var lines = sdpText.split(/\r?\n/);
    var mids = [];
    var i = 0;
    for (i = 0; i < lines.length; i += 1) {
      var line = lines[i];
      if (/^m=video /i.test(line)) {
        var mid = '(no mid)';
        var j = i + 1;
        while (j < lines.length && !/^m=/i.test(lines[j])) {
          var b = /^a=mid:([^\s]+)/i.exec(lines[j]);
          if (b) {
            mid = b[1];
            break;
          }
          j += 1;
        }
        mids.push(mid);
      }
    }
    return mids;
  }

  function refreshMediaDebugPanel() {
    if (!mediaDebugEl) {
      return;
    }
    if (!pc) {
      mediaDebugEl.textContent = '—';
      return;
    }

    function finish(lines) {
      mediaDebugEl.textContent = lines.join('\n');
    }

    var lines = [];
    lines.push('=== Last outbound offer (snapshot at send time) ===');
    lines.push('sent (ISO): ' + (lastOfferSentAtIso || 'n/a'));
    lines.push(
      'recvonly lines requested (transceivers): ' + lastOfferRequestedRecvonlyLines +
        ' | catalog_ids: ' +
        JSON.stringify(lastOfferCatalogIdsSnapshot)
    );
    var lsd = pc.localDescription;
    if (lsd && lsd.sdp) {
      lines.push('local SDP video mids (creation order): ' + sdpVideoRecvonlyMidOrder(lsd.sdp).join(', '));
    }
    lines.push('');

    lines.push('=== ``ontrack`` order (incoming MediaStreamTracks) ===');
    if (!receivedVideoTracksMeta.length) {
      lines.push('(no tracks yet)');
    } else {
      receivedVideoTracksMeta.forEach(function (r, ix) {
        lines.push(
          '  #' + ix + ' mid=' + (r.mid || '?') +
            ' streamId=' + (r.streamId || '?') +
            ' trackId=' + (r.trackId || '?')
        );
      });
    }

    pc.getStats(null).then(function (report) {
      var inbound = [];
      var seenSsrc = {};
      report.forEach(function (s) {
        if (s.type === 'inbound-rtp' && s.kind === 'video' && typeof s.ssrc === 'number') {
          var k = s.ssrc + ':' + String(s.mid || '');
          if (seenSsrc[k]) {
            return;
          }
          seenSsrc[k] = true;
          inbound.push(s);
        }
      });
      inbound.sort(function (a, b) {
        return (a.mid || '').localeCompare(b.mid || '');
      });

      lines.push('');
      lines.push('=== inbound-rtp video (deduped by ssrc+mid) ===');
      if (!inbound.length) {
        lines.push('(no inbound-rtp stats yet)');
      }
      inbound.forEach(function (s, ix) {
        lines.push(
          '  #' + ix +
            ' ssrc=' + s.ssrc +
            ' mid=' + (s.mid || '?') +
            ' mimeType=' + (s.mimeType || '?') +
            ' codecId=' + (s.codecId || '?') +
            ' framesDecoded=' + (typeof s.framesDecoded === 'number' ? s.framesDecoded : '?')
        );
      });

      lines.push('');
      lines.push('=== Peer transceivers (video receivers) ===');
      var txs = pc.getTransceivers();
      var videoTx = txs.filter(function (t) {
        return t.receiver && t.receiver.track && t.receiver.track.kind === 'video';
      });
      lines.push('count=' + videoTx.length);
      videoTx.forEach(function (t, ix) {
        lines.push(
          '  #' + ix + ' mid=' + (t.mid || '?') + ' recv trackId=' + t.receiver.track.id
        );
      });

      var rsd = pc.remoteDescription;
      if (rsd && rsd.sdp) {
        lines.push('');
        lines.push('=== Remote SDP (negotiated ``m=video`` ``a=mid`` order) ===');
        lines.push(String(sdpVideoRecvonlyMidOrder(rsd.sdp).join(', ') || '(none parsed)'));
      }

      finish(lines);
    }).catch(function (e) {
      lines.push('');
      lines.push('getStats() failed: ' + (e && e.message ? e.message : String(e)));
      finish(lines);
    });
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

  function renderCatalogList(ids) {
    if (!catalogListEl) return;
    if (!ids || !ids.length) {
      catalogListEl.textContent = 'Available sensors: waiting for robot hello...';
      return;
    }
    catalogListEl.innerHTML = '';
    var label = document.createElement('div');
    label.textContent = 'Available sensors:';
    catalogListEl.appendChild(label);
    ids.forEach(function (cid) {
      var row = document.createElement('label');
      row.style.display = 'block';
      var cb = document.createElement('input');
      cb.type = 'checkbox';
      cb.setAttribute('data-catalog-id', cid);
      cb.checked = true;
      row.appendChild(cb);
      row.appendChild(document.createTextNode(' ' + cid));
      catalogListEl.appendChild(row);
    });
  }

  /** Matches checked count, or ``1`` recvonly line when none checked (robot picks its default camera). */
  function rtcRecvonlyVideoLineCount() {
    var ids = readCatalogIdsArray();
    return ids.length > 0 ? ids.length : 1;
  }

  function selectedCatalogIdsInOrder() {
    return readCatalogIdsArray();
  }

  /** Checked catalog ids in DOM order. None checked -> ``[]`` (offer omits viewer ids; robot uses its default listing). */
  function readCatalogIdsArray() {
    return selectedCatalogIdsFromCheckboxes();
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
    stopMediaDebugPoll();
    stopGamepadLoop();
    controlDc = null;
    videosEl.innerHTML = '';
    if (streamStatus) {
      streamStatus.textContent = 'Requested ' + numStreams + ' stream(s); negotiating...';
    }
    status.textContent = 'Negotiating WebRTC...';

    receivedVideoTracksMeta = [];
    if (mediaDebugEl) {
      mediaDebugEl.textContent = 'negotiating…';
    }

    pc = new RTCPeerConnection({ iceServers: stunTurnServers });
    controlDc = pc.createDataChannel('krabby-control-v1', { ordered: true });
    controlDc.onopen = function () {
      startGamepadLoop();
    };
    controlDc.onclose = function () {
      stopGamepadLoop();
    };
    pc.addEventListener('connectionstatechange', updateDebug);
    pc.addEventListener('connectionstatechange', function () {
      refreshMediaDebugPanel();
    });
    pc.addEventListener('iceconnectionstatechange', updateDebug);
    pc.addEventListener('signalingstatechange', updateDebug);
    updateDebug();

    var nTracks = 0;
    pc.ontrack = function (ev) {
      receivedVideoTracksMeta.push({
        trackId: ev.track.id,
        streamId: ev.streams[0] ? ev.streams[0].id : '?',
        mid: ev.transceiver && typeof ev.transceiver.mid === 'string' ? ev.transceiver.mid : '(pending)'
      });
      refreshMediaDebugPanel();
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
      lastOfferSentAtIso = new Date().toISOString();
      lastOfferCatalogIdsSnapshot = readCatalogIdsArray().slice();
      lastOfferRequestedRecvonlyLines = numStreams;
      ws.send(JSON.stringify(offerPayload(pc.localDescription.sdp)));
    });
    await pc.setRemoteDescription({ type: 'answer', sdp: ans.sdp });
    status.textContent = 'Playing';
    updateDebug();
    refreshMediaDebugPanel();
    stopMediaDebugPoll();
    mediaDebugPollTimer = setInterval(refreshMediaDebugPanel, 2000);
  }

  function tryStartInitialRtc() {
    if (initialRtcStarted) {
      return;
    }
    if (!ws || ws.readyState !== WebSocket.OPEN) {
      return;
    }
    initialRtcStarted = true;
    if (initialRtcFallbackTimer !== null) {
      clearTimeout(initialRtcFallbackTimer);
      initialRtcFallbackTimer = null;
    }
    startRtc(rtcRecvonlyVideoLineCount()).catch(function (err) {
      status.textContent = 'WebRTC error: ' + (err && err.message ? err.message : String(err));
    });
  }

  function openWebSocket() {
    initialRtcStarted = false;
    if (initialRtcFallbackTimer !== null) {
      clearTimeout(initialRtcFallbackTimer);
      initialRtcFallbackTimer = null;
    }
    ws = new WebSocket(wsUrl);
    ws.onerror = function () {
      status.textContent = 'WebSocket error';
      stopGamepadLoop();
    };
    ws.onclose = function () {
      stopMediaDebugPoll();
      stopGamepadLoop();
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
        tryStartInitialRtc();
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
      status.textContent = 'Waiting for robot hello…';
      try {
        ws.send(JSON.stringify(helloPayload()));
      } catch (e) {}
      initialRtcFallbackTimer = setTimeout(function () {
        initialRtcFallbackTimer = null;
        if (!initialRtcStarted && ws && ws.readyState === WebSocket.OPEN) {
          status.textContent = 'No hello_ack yet; starting WebRTC with current selection…';
          tryStartInitialRtc();
        }
      }, 8000);
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

  var applyCatalogBtn = document.getElementById('applyCatalogStreams');
  if (applyCatalogBtn) {
    applyCatalogBtn.addEventListener('click', function () {
      startRtc(rtcRecvonlyVideoLineCount()).catch(function (err) {
        status.textContent = 'WebRTC error: ' + (err && err.message ? err.message : String(err));
      });
    });
  }

  setInterval(updateControllerDebugPanel, 250);

  boot();
})();
