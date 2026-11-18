# Draft notice to members (send before launch)

> **What the space's Slack bot reads, and what it keeps**
>
> We're turning answers that come up in Slack ("where are the drill bits?",
> "how do I book the laser?") into a searchable knowledge base for members.
> Here is exactly what that means.
>
> **What is read.** Only public channels the bot has been invited to. It
> can't see private channels, DMs or group DMs, and it can't join
> channels by itself. Some channels are excluded even if it's invited.
>
> **How facts are extracted.** A few hours after a thread or conversation
> goes quiet, a model running on the space's own hardware decides whether
> it answers a question about the space, states a fact about it, or points
> to a resource. If so, it writes short standalone facts, such as "The
> drill bits are in the top drawer of the red cabinet."
>
> **What is kept.** Only those facts, a link back to the conversation, and
> the timestamps needed to recheck it. Your messages themselves are not
> stored, and nothing is ever quoted.
>
> **Names are dropped.** Facts never say who said something. Names,
> @handles, emails and phone numbers are removed, and facts about
> individuals, incidents, credentials or admin procedures are thrown out.
>
> **Who can see it.** Members only, never the public chatbot. Facts from
> Slack always rank below the wiki and are labeled as coming from a Slack
> discussion, with the date. Useful ones are proposed for the wiki each
> week.
>
> **Edits and deletions.** Edit or delete your message and the facts drawn
> from it are updated or removed within the hour.
>
> **Keeping a channel out.** Remove the bot from the channel
> (`/remove @bot`); everything derived from that channel is deleted. Or
> ask the admins to add it to the exclusion list.
>
> Questions or concerns: ask in #<admin-channel>.
