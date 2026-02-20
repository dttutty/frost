import os


logger.debug("Process (%s) start..." % os.getpid())
# Only works on Unix/Linux/macOS:
pid = os.fork()
if pid == 0:
    logger.debug(
        "I am child process (%s) and my parent is %s." % (os.getpid(), os.getppid())
    )
else:
    logger.debug("I (%s) just created a child process (%s)." % (os.getpid(), pid))
