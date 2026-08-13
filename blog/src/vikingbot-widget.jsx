import { useEffect } from 'react';

/**
 * Mounts the shared VikingBot embed widget on the blog.
 *
 * Replaces the blog's own 1500-line chat client. The widget renders its own
 * bottom entry pill and right-edge toggle and pushes the shell aside through
 * the rail contract read in themes.css, so nothing is rendered here — the
 * component exists only to own the mount's lifetime.
 *
 * SSG-safe by construction: renderToStaticMarkup never runs effects, so the
 * prerendered HTML carries nothing of the widget and the client mounts it.
 */

const WIDGET_URL =
  import.meta.env.VITE_VIKINGBOT_WIDGET_URL ||
  'https://openviking.net/studio/embed-widget/vikingbot-widget.js';

/** Idle window before the script is fetched; the article must load first. */
const IDLE_FALLBACK_MS = 2500;

let scriptPromise = null;

function loadScriptOnce() {
  if (!scriptPromise) {
    scriptPromise = new Promise((resolve, reject) => {
      const script = document.createElement('script');
      script.src = WIDGET_URL;
      script.async = true;
      script.dataset.vikingbotWidget = '';
      script.addEventListener('load', () => resolve(), { once: true });
      script.addEventListener(
        'error',
        () => {
          script.remove();
          reject(new Error(`Failed to load the VikingBot widget from ${WIDGET_URL}`));
        },
        { once: true },
      );
      document.head.appendChild(script);
    });
  }

  return scriptPromise.catch((error) => {
    // Drop the cached rejection so a later mount can retry the load.
    scriptPromise = null;
    throw error;
  });
}

function whenIdle(run) {
  if (typeof window.requestIdleCallback === 'function') {
    const handle = window.requestIdleCallback(run, { timeout: IDLE_FALLBACK_MS });
    return () => window.cancelIdleCallback(handle);
  }
  const handle = window.setTimeout(run, IDLE_FALLBACK_MS);
  return () => window.clearTimeout(handle);
}

export function VikingBotWidget({ lang }) {
  useEffect(() => {
    let cancelled = false;
    const cancelIdle = whenIdle(() => {
      loadScriptOnce()
        .then(() => {
          if (cancelled) return;
          const widget = window.VikingBotWidget;
          if (!widget) {
            throw new Error('The VikingBot widget script did not register window.VikingBotWidget');
          }
          widget.mount({
            site: 'blog',
            locale: lang === 'zh' ? 'zh' : 'en',
            trigger: 'composer',
            layout: 'push',
            // zouk derives the guest name from this; readers should read as
            // readers, not as the generic 'visitor' every other site sends.
            defaultGuestName: 'reader',
          });
        })
        .catch((error) => {
          // Silent beyond the console: a reader who never asked for the widget
          // should not be shown an error about a script they did not request.
          console.error(error);
        });
    });

    return () => {
      cancelled = true;
      cancelIdle();
      window.VikingBotWidget?.unmount();
    };
  }, [lang]);

  return null;
}

export default VikingBotWidget;
