self.addEventListener('install', () => {
  // Intentionally remain in the normal waiting state when an older worker
  // controls a client. S03 uses this worker only to exercise browser update
  // availability without activating or reloading the running application.
});
