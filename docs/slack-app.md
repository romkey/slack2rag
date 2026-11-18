# Slack app setup

slack2rag reads Slack with a bot token, through the Web API, plus Socket
Mode for live events, so no inbound endpoint is needed.

## Create the app

1. Create an app ("From scratch") at <https://api.slack.com/apps>.
2. **OAuth & Permissions → Bot Token Scopes**, add exactly:
   - `channels:read`: list the channels the bot is in, and the directory
   - `channels:history`: read messages in those channels
   - `pins:read`: pinned messages
   - `bookmarks:read`: channel bookmarks
   - `users:read`: member names, held in memory only, to scrub them from
     facts. Optional: without it, names are still kept out by the
     extraction prompt and the other scrub rules.
3. Install the app to the workspace and copy the **Bot User OAuth Token**
   (`xoxb-…`) into `SLACK_BOT_TOKEN`.
4. For live events: **Socket Mode → Enable**, create an app-level token
   with `connections:write` and put it in `SLACK_APP_TOKEN` (`xapp-…`).
   Then **Event Subscriptions → Subscribe to bot events**:
   `message.channels`, `member_left_channel`, `channel_left`,
   `channel_archive`, `channel_deleted`.

## Never grant

- `channels:join`: the bot must not join channels by itself. The channel
  allowlist is "channels the bot has been invited to", enforced by Slack.
- `groups:history`, `im:history`, `mpim:history` (or their `:read`
  counterparts): no private channels, DMs or group DMs.
- A user token, or a workspace export.

`SLACK_CHANNEL_DENYLIST` is a second guard: channels listed there are never
read, even if someone invites the bot.

## Choosing channels

`/invite @your-bot` in each public channel to index. `/remove @your-bot`
(or archiving the channel) deletes everything derived from it: at once
with live events, otherwise on the next hourly run.
