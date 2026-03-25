(() => {
  const year = document.getElementById("year");
  if (year) year.textContent = String(new Date().getFullYear());

  const btn = document.querySelector("[data-copy]");
  const status = document.querySelector(".copy-status");

  async function copy(sel){
    const el = document.querySelector(sel);
    if (!el) return false;
    const text = (el.innerText || el.textContent || "").trim();
    await navigator.clipboard.writeText(text);
    return true;
  }

  if (btn) {
    btn.addEventListener("click", async () => {
      try{
        const ok = await copy(btn.getAttribute("data-copy"));
        if (status) status.textContent = ok ? "Copied." : "Nothing to copy.";
      } catch(e){
        if (status) status.textContent = "Copy failed (browser blocked clipboard).";
      } finally {
        setTimeout(() => { if (status) status.textContent = ""; }, 1600);
      }
    });
  }

  // Generic helper to synchronize a group of teaser videos within a container
  function initSyncedTeaserSection(container) {
    if (!container) return;

    const videos = Array.from(container.querySelectorAll(".teaser-video"));
    if (!videos.length) return;

    const progressBars = new Map();
    videos.forEach((video) => {
      const parent = video.parentElement;
      if (!parent) return;
      const bar = parent.querySelector(".teaser-progress-bar");
      if (bar) {
        progressBars.set(video, bar);
      }
    });

    // Ensure autoplay-friendly settings
    videos.forEach((video) => {
      try {
        video.muted = true;
        video.playsInline = true;
        video.loop = true;
        // Leave existing preload attribute from HTML, but prefer auto if unset
        if (!video.getAttribute("preload")) {
          video.preload = "auto";
        }
        // Ensure autoplay so that once play() is called, browser can start without user interaction (muted)
        video.setAttribute("autoplay", "autoplay");
      } catch (_) {
        // ignore
      }
    });

    let autoplayStarted = false;

    function startAllVideos() {
      if (!videos.length) return;

      // Reset all videos and progress
      videos.forEach((video) => {
        try {
          video.currentTime = 0;
        } catch (_) {}

        const bar = progressBars.get(video);
        if (bar) {
          bar.style.width = "0%";
        }
      });

      const playPromises = videos.map((video) => {
        try {
          const p = video.play();
          return p && typeof p.then === "function" ? p : Promise.resolve();
        } catch (e) {
          return Promise.reject(e);
        }
      });

      Promise.allSettled(playPromises).then((results) => {
        const anyFulfilled = results.some((r) => r.status === "fulfilled");
        if (anyFulfilled) {
          autoplayStarted = true;
        }
      });
    }

    // Update custom progress bars
    videos.forEach((video) => {
      const bar = progressBars.get(video);
      if (!bar) return;

      video.addEventListener("timeupdate", () => {
        try {
          const duration = video.duration;
          if (!duration || !isFinite(duration)) return;
          const frac = Math.max(0, Math.min(1, video.currentTime / duration));
          bar.style.width = `${frac * 100}%`;
        } catch (_) {
          // ignore
        }
      });
    });

    // Initial attempt when section comes into view (IntersectionObserver)
    if ("IntersectionObserver" in window) {
      const observer = new IntersectionObserver((entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting && !autoplayStarted) {
            startAllVideos();
          }
        });
      }, { threshold: 0.2 });

      observer.observe(container);
    } else {
      // Fallback: start immediately
      startAllVideos();
    }

    // Clicking any video in this section resynchronizes all
    videos.forEach((video) => {
      video.addEventListener("click", (evt) => {
        evt.stopPropagation();
        startAllVideos();
      });
    });
  }

  // Existing teaser grid synchronized autoplay (hero section)
  function initTeaserGridSync() {
    const container = document.getElementById("teaser-grid");
    if (!container) return;

    const videos = Array.from(container.querySelectorAll(".teaser-video"));
    if (!videos.length) return;

    // Map each video to its nearest teaser progress bar (if present)
    const progressBars = new Map();
    videos.forEach((video) => {
      const parent = video.parentElement;
      if (!parent) return;
      const bar = parent.querySelector(".teaser-progress-bar");
      if (bar) {
        progressBars.set(video, bar);
      }
    });

    // Ensure autoplay-friendly settings
    videos.forEach((video) => {
      try {
        video.muted = true;
        video.playsInline = true;
        video.loop = true;
        video.preload = "auto";
      } catch (_) {
        // ignore
      }
    });

    let autoplayStarted = false;

    function startAllTeaserVideos() {
      if (!videos.length) return;

      // Reset all videos to start and reset progress bars
      videos.forEach((video) => {
        try {
          video.currentTime = 0;
        } catch (_) {}

        const bar = progressBars.get(video);
        if (bar) {
          bar.style.width = "0%";
        }
      });

      // Attempt to play all videos nearly simultaneously
      const playPromises = videos.map((video) => {
        try {
          const p = video.play();
          return p && typeof p.then === "function" ? p : Promise.resolve();
        } catch (e) {
          return Promise.reject(e);
        }
      });

      Promise.allSettled(playPromises).then((results) => {
        const allFulfilled = results.every((r) => r.status === "fulfilled");
        if (allFulfilled) {
          autoplayStarted = true;
        }
      });
    }

    // Update custom progress bars using timeupdate
    videos.forEach((video) => {
      const bar = progressBars.get(video);
      if (!bar) return;

      video.addEventListener("timeupdate", () => {
        try {
          const duration = video.duration;
          if (!duration || !isFinite(duration)) return;
          const frac = Math.max(0, Math.min(1, video.currentTime / duration));
          bar.style.width = `${frac * 100}%`;
        } catch (_) {
          // ignore
        }
      });
    });

    // Initial attempt on load
    if (document.readyState === "complete" || document.readyState === "interactive") {
      startAllTeaserVideos();
    } else {
      window.addEventListener("DOMContentLoaded", startAllTeaserVideos, { once: true });
    }

    // Fallback: on first interaction with the teaser grid or window, start/resync
    function onFirstUserInteraction() {
      if (!autoplayStarted) {
        startAllTeaserVideos();
      }
      window.removeEventListener("click", onFirstUserInteraction);
      window.removeEventListener("keydown", onFirstUserInteraction);
    }

    window.addEventListener("click", onFirstUserInteraction);
    window.addEventListener("keydown", onFirstUserInteraction);

    // Clicking any teaser video will resync all videos to start
    videos.forEach((video) => {
      video.addEventListener("click", (evt) => {
        evt.stopPropagation();
        startAllTeaserVideos();
      });
    });
  }

  // Initialize teaser sync after DOM is ready
  function initAllVideoSync() {
    initTeaserGridSync();

    const gtSection = document.getElementById("gt-vs-physvid");
    if (gtSection) {
      initSyncedTeaserSection(gtSection);
    }

    // Any additional comparison sections that use .teaser-video and
    // .teaser-progress-bar inside a container can be wired up here.
    const ablationsSection = document.getElementById("ablations");
    if (ablationsSection) {
      initSyncedTeaserSection(ablationsSection);
    }

    const failuresSection = document.getElementById("failures");
    if (failuresSection) {
      initSyncedTeaserSection(failuresSection);
    }
  }

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", initAllVideoSync);
  } else {
    initAllVideoSync();
  }
})();