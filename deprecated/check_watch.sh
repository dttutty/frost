find /proc/*/fd -lname anon_inode:inotify -printf '%h\n' 2>/dev/null | \
  xargs -I{} sh -c 'cat {}/limits {}/status 2>/dev/null | grep -E "Name|inotify"' | \
  grep -E 'Name|inotify' | less
