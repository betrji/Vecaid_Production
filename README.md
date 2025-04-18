# Vecaid-Beta

Improved GPU Logging
Switched from Tensorflow to Pytorch to streamline GPU and CUDA usage. (Works with my RTX 3060ti)
Frontend is switched from basic HTMl to full React frontend.

Current issues:

Subscribe button doesn't work and tries to look for stripe_price_id or something along those lines. For proper production we need a stripe_webhook, stripe_price_id, and stripe_api_key = sk_live... (secret key) in the backend and stripe_api_key = pk_live... (public key) in the frontend. Integrate however you see fit.

Whenever the /api/info route is called in the frontend after entering in a ticker symbol in the chatbox, a 404 error gets returned. The /api route probably needs to be properly registered in both the frontend and the backend files.

Review what needs to be added, compare the files in backendPastModels and the current file in MainBackend.
