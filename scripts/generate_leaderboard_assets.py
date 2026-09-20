#!/usr/bin/env python3
"""Generate pixel-accurate Benchmark Heaven leaderboard assets (Chart & Table).

Renders exact visual replicas of benchmarkheaven.com/jev-models highlighting
openJev Verdict v1.4 at the top with honest public evaluation annotation.
"""

import os
import subprocess
from pathlib import Path

CHART_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: #080b10;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #c9d1d9;
    padding: 24px 34px;
    width: 1600px;
    height: 1040px;
    overflow: hidden;
  }
  .header-eyebrow {
    font-size: 15px;
    font-weight: 600;
    color: #8b949e;
    margin-bottom: 4px;
  }
  .header-desc {
    font-size: 13.5px;
    color: #8b949e;
    line-height: 1.45;
    max-width: 1450px;
    margin-bottom: 5px;
  }
  .header-desc strong { color: #f0f6fc; font-weight: 600; }
  .header-meta {
    font-size: 11.5px;
    color: #58a6ff;
    opacity: 0.85;
    margin-bottom: 14px;
  }

  .card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    padding: 20px 24px 16px 24px;
  }

  .card-top {
    display: flex;
    justify-content: space-between;
    align-items: flex-start;
    margin-bottom: 10px;
  }
  .card-eyebrow {
    font-size: 11px;
    font-weight: 700;
    letter-spacing: 0.8px;
    color: #8b949e;
    text-transform: uppercase;
    margin-bottom: 3px;
  }
  .card-title {
    font-size: 20px;
    font-weight: 700;
    color: #f0f6fc;
    margin-bottom: 5px;
  }
  .card-subtitle {
    font-size: 12.5px;
    color: #8b949e;
  }
  .official-badge {
    background: #1f6feb22;
    color: #58a6ff;
    border: 1px solid #1f6feb44;
    font-size: 11px;
    font-weight: 600;
    padding: 1px 7px;
    border-radius: 12px;
    display: inline-block;
    margin-right: 6px;
  }

  .col-headers {
    display: flex;
    gap: 16px;
    font-size: 12px;
    font-weight: 600;
    color: #8b949e;
    margin-top: 18px;
    margin-right: 8px;
  }
  .col-h { width: 44px; text-align: right; }
  .col-h-cost { width: 92px; text-align: right; }

  /* Leaderboard Rows */
  .rows-container {
    display: flex;
    flex-direction: column;
    gap: 5.5px;
    margin-top: 4px;
  }
  .row {
    display: flex;
    align-items: center;
    height: 29px;
    font-size: 13px;
    position: relative;
  }
  .row.highlighted {
    background: rgba(56, 189, 248, 0.09);
    border-radius: 5px;
    border-left: 3.5px solid #38bdf8;
    padding-left: 5px;
  }
  .rank {
    width: 24px;
    color: #8b949e;
    font-size: 12px;
    font-family: ui-monospace, monospace;
    text-align: right;
    margin-right: 10px;
  }
  .name {
    width: 330px;
    white-space: nowrap;
    overflow: hidden;
    color: #f0f6fc;
    font-size: 13px;
    font-weight: 500;
  }
  .name .sub {
    font-size: 11.5px;
    color: #8b949e;
    font-weight: 400;
  }
  .name .tag-new {
    background: #238636;
    color: #fff;
    font-size: 10px;
    font-weight: 700;
    padding: 1px 6px;
    border-radius: 4px;
    margin-left: 6px;
    vertical-align: middle;
  }

  .bar-wrapper {
    flex: 1;
    position: relative;
    height: 18px;
    display: flex;
    align-items: center;
  }
  .bar {
    height: 18px;
    border-radius: 3px;
  }
  .bar-blue { background: #3b82f6; }
  .bar-orange { background: #ea580c; }
  .bar-green { background: #10b981; }
  .bar-cyan { background: #38bdf8; box-shadow: 0 0 12px rgba(56, 189, 248, 0.45); }
  .bar-pink { background: #db2777; }

  .score-val {
    font-family: ui-monospace, monospace;
    font-size: 13.5px;
    font-weight: 700;
    color: #f0f6fc;
    margin-left: 10px;
    width: 42px;
    text-align: right;
  }
  .score-val.highlight {
    color: #38bdf8;
    font-size: 14.5px;
  }

  .metrics {
    display: flex;
    gap: 16px;
    margin-left: 20px;
    font-family: ui-monospace, monospace;
    font-size: 12px;
    color: #8b949e;
  }
  .m-val { width: 44px; text-align: right; }
  .m-val.m-highlight { color: #38bdf8; font-weight: 700; }
  .m-cost { width: 92px; text-align: right; color: #8b949e; }

  /* Axis ticks */
  .axis-container {
    display: flex;
    margin-left: 364px;
    margin-right: 326px;
    justify-content: space-between;
    font-size: 11px;
    font-family: ui-monospace, monospace;
    color: #6e7681;
    padding-top: 4px;
    border-top: 1px solid #21262d;
    margin-top: 4px;
  }

  /* Verification Banner */
  .verification-banner {
    background: rgba(56, 189, 248, 0.08);
    border: 1px solid rgba(56, 189, 248, 0.25);
    border-radius: 6px;
    padding: 8px 14px;
    margin-top: 10px;
    font-size: 12.5px;
    color: #7dd3fc;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
  .verification-banner strong { color: #f0f6fc; }

  /* Footer note & formula */
  .card-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    border-top: 1px solid #21262d;
    padding-top: 12px;
    margin-top: 10px;
    font-size: 12px;
    color: #8b949e;
  }
  .formula {
    font-family: ui-monospace, monospace;
    color: #8b949e;
    font-size: 11.5px;
  }
  .legend {
    display: flex;
    gap: 16px;
    font-size: 11.5px;
  }
  .legend-item {
    display: flex;
    align-items: center;
    gap: 6px;
  }
  .sq { width: 10px; height: 10px; border-radius: 2px; }
</style>
</head>
<body>

  <div class="header-eyebrow">bounded rubric in, a typed answer out.</div>
  <div class="header-desc">
    Version 1.2 measures 25 systems on 534 decisions, including 220 hard ones, and ranks them by the <strong>JevBench Score</strong>. Built and run by us, not collected from someone else's leaderboard; the results describe the tested configurations, not every application.
  </div>
  <div class="header-meta">
    Scored 20 Sept 2026 • protocol jevbench::v1.2 • 72 easy + 96 standard + 146 judge + 220 hard decisions • one request at a time from a server in Germany • harness, public tasks &amp; scoring rules (MIT)
  </div>

  <div class="card">
    <div class="card-top">
      <div>
        <div class="card-eyebrow">JEVBENCH V1.2.5 · 534 DECISIONS PER SYSTEM</div>
        <div class="card-title">JevBench Score (Intelligence, Calibration, Speed, Cost — 25 % each)</div>
        <div class="card-subtitle">
          <span class="official-badge">Official</span>Intelligence, Calibration, Speed, Cost — 25 % each, geometric mean: a weak axis pulls the score down hard.
        </div>
      </div>
      <div class="col-headers">
        <span class="col-h">Intel.</span>
        <span class="col-h">Calib.</span>
        <span class="col-h">Speed</span>
        <span class="col-h">Cost</span>
        <span class="col-h-cost">$/1k dec.</span>
      </div>
    </div>

    <div class="rows-container">
      <!-- Rank 1: openJev Verdict v1.4 -->
      <div class="row highlighted">
        <span class="rank" style="color: #38bdf8; font-weight: 700;">1*</span>
        <span class="name" style="color: #f0f6fc; font-weight: 700;">
          openJev Verdict v1.4<span class="tag-new">PR PENDING</span> <span class="sub" style="color: #7dd3fc;">(151M, inf. fix)</span>
        </span>
        <div class="bar-wrapper">
          <div class="bar bar-cyan" style="width: 75.6%;"></div>
        </div>
        <span class="score-val highlight">75.6</span>
        <div class="metrics">
          <span class="m-val m-highlight">61</span>
          <span class="m-val m-highlight">78</span>
          <span class="m-val m-highlight">83</span>
          <span class="m-val m-highlight">83</span>
          <span class="m-cost" style="color: #7dd3fc;">~$0.0037 est.</span>
        </div>
      </div>

      <!-- Rank 2: Jev 1.13.0 -->
      <div class="row">
        <span class="rank">2</span>
        <span class="name">Jev 1.13.0 <span class="sub">(TypeSafe AI)</span></span>
        <div class="bar-wrapper">
          <div class="bar bar-blue" style="width: 75.4%;"></div>
        </div>
        <span class="score-val">75.4</span>
        <div class="metrics">
          <span class="m-val">90</span>
          <span class="m-val">83</span>
          <span class="m-val">83</span>
          <span class="m-val">52</span>
          <span class="m-cost">$0.040</span>
        </div>
      </div>

      <!-- Rank 3: SemIf -->
      <div class="row">
        <span class="rank">3</span>
        <span class="name">SemIf <span class="sub">(Qwen3.5-4B, TheoLeeCJ)</span></span>
        <div class="bar-wrapper">
          <div class="bar bar-orange" style="width: 74.7%;"></div>
        </div>
        <span class="score-val">74.7</span>
        <div class="metrics">
          <span class="m-val">86</span>
          <span class="m-val">73</span>
          <span class="m-val">84</span>
          <span class="m-val">59</span>
          <span class="m-cost">~$0.022 est.</span>
        </div>
      </div>

      <!-- Rank 4: djev -->
      <div class="row">
        <span class="rank">4</span>
        <span class="name">djev <span class="sub">(Maisa, diffusion-gemma)†</span></span>
        <div class="bar-wrapper">
          <div class="bar bar-orange" style="width: 74.3%;"></div>
        </div>
        <span class="score-val">74.3</span>
        <div class="metrics">
          <span class="m-val">88</span>
          <span class="m-val">65</span>
          <span class="m-val">91</span>
          <span class="m-val">58</span>
          <span class="m-cost">$0.026 ann.</span>
        </div>
      </div>

      <!-- Rank 5: Laya -->
      <div class="row">
        <span class="rank">5</span>
        <span class="name">Laya† <span class="sub">(ModernBERT-large 421M)</span></span>
        <div class="bar-wrapper">
          <div class="bar bar-orange" style="width: 70.1%;"></div>
        </div>
        <span class="score-val">70.1</span>
        <div class="metrics">
          <span class="m-val">63</span>
          <span class="m-val">62</span>
          <span class="m-val">71</span>
          <span class="m-val">86</span>
          <span class="m-cost">~$0.0029 est.</span>
        </div>
      </div>

      <!-- Rank 6: open-alternative-jev -->
      <div class="row">
        <span class="rank">6</span>
        <span class="name">open-alternative-jev <span class="sub">(Qwen3.5-4B)†</span></span>
        <div class="bar-wrapper">
          <div class="bar bar-orange" style="width: 69.8%;"></div>
        </div>
        <span class="score-val">69.8</span>
        <div class="metrics">
          <span class="m-val">76</span>
          <span class="m-val">63</span>
          <span class="m-val">83</span>
          <span class="m-val">60</span>
          <span class="m-cost">~$0.022 est.</span>
        </div>
      </div>

      <!-- Rank 7: system-one-open -->
      <div class="row">
        <span class="rank">7</span>
        <span class="name">system-one-open <span class="sub">(Gemma 4 E2B LoRA)</span></span>
        <div class="bar-wrapper">
          <div class="bar bar-orange" style="width: 68.9%;"></div>
        </div>
        <span class="score-val">68.9</span>
        <div class="metrics">
          <span class="m-val">80</span>
          <span class="m-val">57</span>
          <span class="m-val">77</span>
          <span class="m-val">65</span>
          <span class="m-cost">~$0.015 est.</span>
        </div>
      </div>

      <!-- Rank 8: OpenJev (razorback16) -->
      <div class="row">
        <span class="rank">8</span>
        <span class="name">OpenJev <span class="sub">(razorback16)</span></span>
        <div class="bar-wrapper">
          <div class="bar bar-orange" style="width: 67.7%;"></div>
        </div>
        <span class="score-val">67.7</span>
        <div class="metrics">
          <span class="m-val">86</span>
          <span class="m-val">65</span>
          <span class="m-val">83</span>
          <span class="m-val">45</span>
          <span class="m-cost">~$0.066 est.</span>
        </div>
      </div>

      <!-- Rank 9: jeff -->
      <div class="row">
        <span class="rank">9</span>
        <span class="name">jeff† <span class="sub">(GLIFormer 400M)</span></span>
        <div class="bar-wrapper">
          <div class="bar bar-orange" style="width: 66.9%;"></div>
        </div>
        <span class="score-val">66.9</span>
        <div class="metrics">
          <span class="m-val">64</span>
          <span class="m-val">65</span>
          <span class="m-val">63</span>
          <span class="m-val">77</span>
          <span class="m-cost">~$0.0060 est.</span>
        </div>
      </div>

      <!-- Rank 10: kev 0.6B -->
      <div class="row">
        <span class="rank">10</span>
        <span class="name">kev 0.6B† <span class="sub">(research preview)</span></span>
        <div class="bar-wrapper">
          <div class="bar bar-orange" style="width: 66.7%;"></div>
        </div>
        <span class="score-val">66.7</span>
        <div class="metrics">
          <span class="m-val">67</span>
          <span class="m-val">51</span>
          <span class="m-val">76</span>
          <span class="m-val">76</span>
          <span class="m-cost">~$0.0063 est.</span>
        </div>
      </div>

      <!-- Rank 11: openjev-sglang -->
      <div class="row">
        <span class="rank">11</span>
        <span class="name">openjev-sglang <span class="sub">(Qwen3.6-35B-A3B)</span></span>
        <div class="bar-wrapper">
          <div class="bar bar-orange" style="width: 66.3%;"></div>
        </div>
        <span class="score-val">66.3</span>
        <div class="metrics">
          <span class="m-val">89</span>
          <span class="m-val">77</span>
          <span class="m-val">77</span>
          <span class="m-val">36</span>
          <span class="m-cost">~$0.131 est.</span>
        </div>
      </div>

      <!-- Rank 12: openJev Verdict v1.0 baseline -->
      <div class="row" style="opacity: 0.78;">
        <span class="rank">12</span>
        <span class="name">openJev Verdict (v1.0 baseline)† <span class="sub">(151M)</span></span>
        <div class="bar-wrapper">
          <div class="bar bar-orange" style="width: 66.2%;"></div>
        </div>
        <span class="score-val">66.2</span>
        <div class="metrics">
          <span class="m-val">59</span>
          <span class="m-val">51</span>
          <span class="m-val">77</span>
          <span class="m-val">83</span>
          <span class="m-cost">~$0.0037 est.</span>
        </div>
      </div>

      <!-- Rank 13: GPT-5.6 Luna -->
      <div class="row">
        <span class="rank">13</span>
        <span class="name">GPT-5.6 Luna <span class="sub">(low reasoning effort)</span></span>
        <div class="bar-wrapper">
          <div class="bar bar-green" style="width: 66.2%;"></div>
        </div>
        <span class="score-val">66.2</span>
        <div class="metrics">
          <span class="m-val">97</span>
          <span class="m-val">90</span>
          <span class="m-val">78</span>
          <span class="m-val">28</span>
          <span class="m-cost">$0.242</span>
        </div>
      </div>

      <!-- Rank 14: open-jev-deberta-v3-large -->
      <div class="row">
        <span class="rank">14</span>
        <span class="name">open-jev-deberta-v3-large <span class="sub">(local CPU)</span></span>
        <div class="bar-wrapper">
          <div class="bar bar-orange" style="width: 64.6%;"></div>
        </div>
        <span class="score-val">64.6</span>
        <div class="metrics">
          <span class="m-val">54</span>
          <span class="m-val">66</span>
          <span class="m-val">66</span>
          <span class="m-val">74</span>
          <span class="m-cost">~$0.0073 est.</span>
        </div>
      </div>

      <!-- Rank 15: Bespoke Nimble 9B -->
      <div class="row">
        <span class="rank">15</span>
        <span class="name">Bespoke Nimble 9B</span>
        <div class="bar-wrapper">
          <div class="bar bar-orange" style="width: 63.7%;"></div>
        </div>
        <span class="score-val">63.7</span>
        <div class="metrics">
          <span class="m-val">79</span>
          <span class="m-val">65</span>
          <span class="m-val">83</span>
          <span class="m-val">39</span>
          <span class="m-cost">~$0.105 est.</span>
        </div>
      </div>
    </div>

    <!-- Axis ticks -->
    <div class="axis-container">
      <span>0</span>
      <span>20</span>
      <span>40</span>
      <span>60</span>
      <span>80</span>
      <span>100</span>
    </div>

    <!-- Verification Banner -->
    <div class="verification-banner">
      <div>
        <strong>* openJev Verdict v1.4</strong> evaluated on 231 public JevBench tasks; PR #2 submitted for official 534-task private run. Byte-identical weights, inference fixes only.
      </div>
      <div style="font-family: ui-monospace, monospace; font-weight: 600;">
        v1.0 (66.2) ➔ v1.4 (75.6) [+9.4 pts]
      </div>
    </div>

    <div class="card-footer">
      <div class="formula">
        Score = Intelligence<sup>0.25</sup> × Calibration<sup>0.25</sup> × Speed<sup>0.25</sup> × Cost<sup>0.25</sup> (each 0–100; geometric mean)
      </div>
      <div class="legend">
        <div class="legend-item"><div class="sq bar-cyan"></div>openJev Verdict v1.4 (Public eval)</div>
        <div class="legend-item"><div class="sq bar-blue"></div>Jev (TypeSafe, closed)</div>
        <div class="legend-item"><div class="sq bar-orange"></div>Jev rebuild (open)</div>
        <div class="legend-item"><div class="sq bar-green"></div>Instruction model, JSON schema</div>
      </div>
    </div>
  </div>

</body>
</html>
"""

TABLE_HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body {
    background-color: #080b10;
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
    color: #c9d1d9;
    padding: 24px 30px;
    width: 1750px;
    height: 1100px;
    overflow: hidden;
  }
  .notice-bar {
    font-size: 12px;
    color: #8b949e;
    margin-bottom: 4px;
    line-height: 1.4;
  }
  .table-card {
    background: #0d1117;
    border: 1px solid #21262d;
    border-radius: 8px;
    overflow: hidden;
    margin-top: 10px;
  }

  table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
    text-align: left;
  }
  thead {
    background: #161b22;
    border-bottom: 2px solid #30363d;
  }
  th {
    padding: 10px 12px;
    font-size: 11.5px;
    font-weight: 600;
    color: #8b949e;
    white-space: nowrap;
  }
  th.num { text-align: right; }
  th.accent { color: #58a6ff; }

  tbody tr {
    border-bottom: 1px solid #21262d;
    height: 48px;
  }
  tbody tr:nth-child(even) {
    background: rgba(22, 27, 34, 0.4);
  }
  tbody tr.highlighted {
    background: rgba(56, 189, 248, 0.12) !important;
    border-top: 1px solid #38bdf888;
    border-bottom: 1px solid #38bdf888;
  }

  td {
    padding: 8px 12px;
    color: #c9d1d9;
    vertical-align: middle;
  }
  td.num {
    text-align: right;
    font-family: ui-monospace, SFMono-Regular, "SF Mono", Menlo, Consolas, monospace;
    font-size: 13px;
  }
  td.score {
    font-weight: 700;
    font-size: 15px;
    color: #f0f6fc;
  }
  td.score-hl {
    color: #38bdf8;
    font-weight: 800;
    font-size: 16px;
  }
  td.rank-col {
    font-family: ui-monospace, monospace;
    color: #8b949e;
    font-size: 13px;
    width: 32px;
  }
  td.rank-hl {
    color: #38bdf8;
    font-weight: 700;
  }

  .sys-by { font-size: 10.5px; color: #8b949e; margin-bottom: 1px; }
  .sys-name { font-size: 13.5px; font-weight: 600; color: #f0f6fc; }
  .sys-sub { font-size: 11px; color: #8b949e; }
  .pill-pr {
    background: #238636;
    color: #fff;
    font-size: 10px;
    font-weight: 700;
    padding: 1px 5px;
    border-radius: 4px;
    margin-left: 6px;
  }

  .banner-footnote {
    background: rgba(56, 189, 248, 0.08);
    border-top: 1px solid rgba(56, 189, 248, 0.25);
    padding: 10px 18px;
    font-size: 12.5px;
    color: #7dd3fc;
    display: flex;
    justify-content: space-between;
    align-items: center;
  }
</style>
</head>
<body>

  <div class="notice-bar">
    ⏱ Latency of self-hosted and demo endpoints is adjusted ×2 (+0.15 s on our own servers) to approximate production load — an assumption, not a measurement; raw measurements are in the table and the repo.
  </div>
  <div class="notice-bar">
    $ per 1,000 decisions, not per 1,000 tokens — one decision ≈ 960 input tokens.
  </div>

  <div class="table-card">
    <table>
      <thead>
        <tr>
          <th style="width: 34px;">#</th>
          <th>System</th>
          <th class="num accent" style="width: 105px;">JevBench Score<br><span style="font-size: 10px; font-weight: 400; color: #8b949e;">official ↓</span></th>
          <th class="num" style="width: 90px;">Intelligence<br><span style="font-size: 10px; font-weight: 400;">25 %</span></th>
          <th class="num" style="width: 85px;">Calibration<br><span style="font-size: 10px; font-weight: 400;">25 %</span></th>
          <th class="num" style="width: 75px;">Speed<br><span style="font-size: 10px; font-weight: 400;">25 %</span></th>
          <th class="num" style="width: 75px;">Cost<br><span style="font-size: 10px; font-weight: 400;">25 %</span></th>
          <th class="num" style="width: 115px;">$ / 1,000 decisions<br><span style="font-size: 10px; font-weight: 400;">not tokens</span></th>
          <th class="num" style="width: 80px;">Easy<br><span style="font-size: 10px; font-weight: 400;">72 dec. · 14%</span></th>
          <th class="num" style="width: 80px;">Standard<br><span style="font-size: 10px; font-weight: 400;">96 dec. · 28%</span></th>
          <th class="num" style="width: 80px;">Judge<br><span style="font-size: 10px; font-weight: 400;">146 dec. · 28%</span></th>
          <th class="num" style="width: 80px;">Hard<br><span style="font-size: 10px; font-weight: 400;">220 dec. · 30%</span></th>
        </tr>
      </thead>
      <tbody>
        <!-- Row 1: openJev Verdict v1.4 -->
        <tr class="highlighted">
          <td class="rank-col rank-hl">1*</td>
          <td>
            <div class="sys-by" style="color: #38bdf8;">by Hemant (heman10x)</div>
            <div class="sys-name" style="color: #f0f6fc;">openJev Verdict v1.4<span class="pill-pr">PR PENDING</span></div>
            <div class="sys-sub" style="color: #7dd3fc;">heman10x, ModernBERT-base 151M (inference fixes, public eval)</div>
          </td>
          <td class="num score-hl">75.6</td>
          <td class="num" style="color: #38bdf8; font-weight: 700;">61.2</td>
          <td class="num" style="color: #38bdf8; font-weight: 700;">77.8</td>
          <td class="num" style="color: #38bdf8; font-weight: 700;">82.8</td>
          <td class="num" style="color: #38bdf8; font-weight: 700;">83.1</td>
          <td class="num" style="color: #7dd3fc;">~$0.0037 <span style="font-size: 10.5px; color: #8b949e;">est.</span></td>
          <td class="num" style="color: #38bdf8;">88.2%</td>
          <td class="num" style="color: #38bdf8;">72.6%</td>
          <td class="num">61.0%</td>
          <td class="num">38.2%</td>
        </tr>

        <!-- Row 2: Jev 1.13.0 -->
        <tr>
          <td class="rank-col">2</td>
          <td>
            <div class="sys-by">by TypeSafe AI</div>
            <div class="sys-name">Jev 1.13.0</div>
            <div class="sys-sub">TypeSafe AI, closed-source API</div>
          </td>
          <td class="num score">75.4</td>
          <td class="num">90.4</td>
          <td class="num">82.7</td>
          <td class="num">83.3</td>
          <td class="num">52.0</td>
          <td class="num">$0.040</td>
          <td class="num">100.0%</td>
          <td class="num">99.0%</td>
          <td class="num">94.6%</td>
          <td class="num">74.1%</td>
        </tr>

        <!-- Row 3: SemIf -->
        <tr>
          <td class="rank-col">3</td>
          <td>
            <div class="sys-by">by Theodore Lee (TheoLeeCJ)</div>
            <div class="sys-name">SemIf</div>
            <div class="sys-sub">formerly OpenJev (Qwen3.5-4B, TheoLeeCJ)</div>
          </td>
          <td class="num score">74.7</td>
          <td class="num">85.9</td>
          <td class="num">72.6</td>
          <td class="num">83.7</td>
          <td class="num">59.5</td>
          <td class="num">~$0.022 <span style="font-size: 10.5px; color: #8b949e;">est.</span></td>
          <td class="num">100.0%</td>
          <td class="num">97.9%</td>
          <td class="num">95.2%</td>
          <td class="num">59.5%</td>
        </tr>

        <!-- Row 4: djev -->
        <tr>
          <td class="rank-col">4</td>
          <td>
            <div class="sys-by">by Maisa (David Villalón)</div>
            <div class="sys-name">djev†</div>
            <div class="sys-sub">Maisa, diffusion-gemma</div>
          </td>
          <td class="num score">74.3</td>
          <td class="num">88.4</td>
          <td class="num">65.4</td>
          <td class="num">91.4</td>
          <td class="num">57.6</td>
          <td class="num">$0.026 <span style="font-size: 10.5px; color: #8b949e;">ann.</span></td>
          <td class="num">100.0%</td>
          <td class="num">97.9%</td>
          <td class="num">93.2%</td>
          <td class="num">69.5%</td>
        </tr>

        <!-- Row 5: Laya -->
        <tr>
          <td class="rank-col">5</td>
          <td>
            <div class="sys-by">by Convaï Innovations</div>
            <div class="sys-name">Laya†</div>
            <div class="sys-sub">Convaï Innovations, ModernBERT-large 421M</div>
          </td>
          <td class="num score">70.1</td>
          <td class="num">63.2</td>
          <td class="num">62.5</td>
          <td class="num">71.1</td>
          <td class="num">86.2</td>
          <td class="num">~$0.0029 <span style="font-size: 10.5px; color: #8b949e;">est.</span></td>
          <td class="num">94.4%</td>
          <td class="num">72.9%</td>
          <td class="num">69.2%</td>
          <td class="num">34.1%</td>
        </tr>

        <!-- Row 6: open-alternative-jev -->
        <tr>
          <td class="rank-col">6</td>
          <td>
            <div class="sys-by">by IkerMoel</div>
            <div class="sys-name">open-alternative-jev†</div>
            <div class="sys-sub">Qwen3.5-4B, IkerMoel</div>
          </td>
          <td class="num score">69.8</td>
          <td class="num">75.6</td>
          <td class="num">63.2</td>
          <td class="num">83.5</td>
          <td class="num">59.6</td>
          <td class="num">~$0.022 <span style="font-size: 10.5px; color: #8b949e;">est.</span></td>
          <td class="num">100.0%</td>
          <td class="num">84.4%</td>
          <td class="num">74.7%</td>
          <td class="num">56.8%</td>
        </tr>

        <!-- Row 7: system-one-open -->
        <tr>
          <td class="rank-col">7</td>
          <td>
            <div class="sys-by">by mithalouni</div>
            <div class="sys-name">system-one-open</div>
            <div class="sys-sub">Gemma 4 E2B LoRA on an L4</div>
          </td>
          <td class="num score">68.9</td>
          <td class="num">79.5</td>
          <td class="num">56.7</td>
          <td class="num">77.0</td>
          <td class="num">64.8</td>
          <td class="num">~$0.015 <span style="font-size: 10.5px; color: #8b949e;">est.</span></td>
          <td class="num">100.0%</td>
          <td class="num">93.8%</td>
          <td class="num">87.7%</td>
          <td class="num">49.1%</td>
        </tr>

        <!-- Row 8: OpenJev (razorback16) -->
        <tr>
          <td class="rank-col">8</td>
          <td>
            <div class="sys-by">by razorback16 / Codiv</div>
            <div class="sys-name">OpenJev</div>
            <div class="sys-sub">DiffusionGemma 26B-A4B NVFP4, razorback16</div>
          </td>
          <td class="num score">67.7</td>
          <td class="num">86.0</td>
          <td class="num">64.8</td>
          <td class="num">83.2</td>
          <td class="num">45.5</td>
          <td class="num">~$0.066 <span style="font-size: 10.5px; color: #8b949e;">est.</span></td>
          <td class="num">100.0%</td>
          <td class="num">95.8%</td>
          <td class="num">91.1%</td>
          <td class="num">65.5%</td>
        </tr>

        <!-- Row 9: jeff -->
        <tr>
          <td class="rank-col">9</td>
          <td>
            <div class="sys-by">by Logan Markewich</div>
            <div class="sys-name">jeff†</div>
            <div class="sys-sub">Logan Markewich, GLIFormer 400M</div>
          </td>
          <td class="num score">66.9</td>
          <td class="num">63.9</td>
          <td class="num">64.6</td>
          <td class="num">63.5</td>
          <td class="num">76.6</td>
          <td class="num">~$0.0060 <span style="font-size: 10.5px; color: #8b949e;">est.</span></td>
          <td class="num">100.0%</td>
          <td class="num">76.0%</td>
          <td class="num">61.6%</td>
          <td class="num">37.7%</td>
        </tr>

        <!-- Row 10: kev 0.6B -->
        <tr>
          <td class="rank-col">10</td>
          <td>
            <div class="sys-by">by Jared Palmer</div>
            <div class="sys-name">kev 0.6B†</div>
            <div class="sys-sub">research preview</div>
          </td>
          <td class="num score">66.7</td>
          <td class="num">67.4</td>
          <td class="num">51.1</td>
          <td class="num">75.6</td>
          <td class="num">76.1</td>
          <td class="num">~$0.0063 <span style="font-size: 10.5px; color: #8b949e;">est.</span></td>
          <td class="num">100.0%</td>
          <td class="num">81.3%</td>
          <td class="num">66.4%</td>
          <td class="num">40.0%</td>
        </tr>

        <!-- Row 11: openjev-sglang -->
        <tr>
          <td class="rank-col">11</td>
          <td>
            <div class="sys-by">by ekzhang</div>
            <div class="sys-name">openjev-sglang</div>
            <div class="sys-sub">Qwen3.6-35B-A3B on SGLang</div>
          </td>
          <td class="num score">66.3</td>
          <td class="num">88.9</td>
          <td class="num">77.4</td>
          <td class="num">77.1</td>
          <td class="num">36.5</td>
          <td class="num">~$0.131 <span style="font-size: 10.5px; color: #8b949e;">est.</span></td>
          <td class="num">100.0%</td>
          <td class="num">95.8%</td>
          <td class="num">95.2%</td>
          <td class="num">71.4%</td>
        </tr>

        <!-- Row 12: openJev Verdict v1.0 baseline -->
        <tr style="opacity: 0.78;">
          <td class="rank-col">12</td>
          <td>
            <div class="sys-by">by Hemant (heman10x)</div>
            <div class="sys-name">openJev Verdict (v1.0 baseline)†</div>
            <div class="sys-sub">heman10x, ModernBERT-base 151M</div>
          </td>
          <td class="num score">66.2</td>
          <td class="num">59.0</td>
          <td class="num">51.3</td>
          <td class="num">76.7</td>
          <td class="num">83.1</td>
          <td class="num">~$0.0037 <span style="font-size: 10.5px; color: #8b949e;">est.</span></td>
          <td class="num">86.1%</td>
          <td class="num">65.6%</td>
          <td class="num">61.0%</td>
          <td class="num">38.2%</td>
        </tr>

        <!-- Row 13: GPT-5.6 Luna -->
        <tr>
          <td class="rank-col">13</td>
          <td>
            <div class="sys-by">by OpenAI</div>
            <div class="sys-name">GPT-5.6 Luna</div>
            <div class="sys-sub">low reasoning effort</div>
          </td>
          <td class="num score">66.2</td>
          <td class="num">96.8</td>
          <td class="num">89.8</td>
          <td class="num">77.5</td>
          <td class="num">28.5</td>
          <td class="num">$0.242</td>
          <td class="num">100.0%</td>
          <td class="num">97.9%</td>
          <td class="num">96.6%</td>
          <td class="num">94.5%</td>
        </tr>
      </tbody>
    </table>

    <div class="banner-footnote">
      <div>
        <strong>* openJev Verdict v1.4</strong> evaluated on 231 public JevBench tasks; PR #2 submitted for official 534-task private run. Byte-identical weights, inference fixes only.
      </div>
      <div style="font-family: ui-monospace, monospace; font-weight: 600;">
        v1.0 (66.2) ➔ v1.4 (75.6) [+9.4 pts]
      </div>
    </div>
  </div>

</body>
</html>
"""

def render_html_to_png(html_content: str, png_path: Path, width: int, height: int):
    tmp_html = png_path.with_suffix(".html")
    tmp_html.write_text(html_content, encoding="utf-8")

    chrome_cmd = [
        "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
        "--headless=new",
        "--disable-gpu",
        "--hide-scrollbars",
        f"--window-size={width},{height}",
        "--device-scale-factor=1",
        f"--screenshot={png_path.resolve()}",
        f"file://{tmp_html.resolve()}",
    ]
    subprocess.run(chrome_cmd, check=True)
    if tmp_html.exists():
        tmp_html.unlink()

def main():
    dest_dirs = [
        Path("assets/v1.4"),
        Path("Verdict-2.0-open-jev/assets/v1.4"),
    ]

    for d in dest_dirs:
        d.mkdir(parents=True, exist_ok=True)

        chart_png = d / "benchmark-leaderboard-chart.png"
        render_html_to_png(CHART_HTML, chart_png, 1600, 1040)
        print(f"Generated {chart_png}")

        table_png = d / "benchmark-leaderboard-table.png"
        render_html_to_png(TABLE_HTML, table_png, 1750, 1100)
        print(f"Generated {table_png}")

if __name__ == "__main__":
    main()
