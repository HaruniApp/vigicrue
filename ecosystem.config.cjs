module.exports = {
  apps: [{
    name: 'vilaine-amont-api',
    script: 'backend/src/index.js',
    env: {
      NODE_ENV: 'production',
      PORT: 3001,
    },
  }],
};
