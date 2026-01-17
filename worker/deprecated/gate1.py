import threading


class TargetNumberGate:
    """
    A gate that only allows a thread with a specific target number to pass.
    It uses a Condition variable to manage waiting and notification.
    """

    def __init__(self, allow_broadcast: bool = True):
        # The condition object bundles a lock with wait/notify capabilities.
        self.condition = threading.Condition()
        self.allow_broadcast = allow_broadcast

    def wait_for_turn(self, is_broadcast: bool):
        """
        A thread calls this method to attempt entry.
        It will wait until its number matches the gate's target number.
        """
        with self.condition:
            # The 'while' loop is crucial. It prevents 'spurious wakeups'.
            # A thread must re-check the condition after being woken up.
            while is_broadcast != self.allow_broadcast:
                print("尝试进入，开始等待...")
                # self.condition.wait() automatically releases the lock
                # and puts the thread to sleep until notified.
                self.condition.wait()
            self.allow_broadcast = not self.allow_broadcast

    def open_for(self, new_target: int):
        """
        The main thread calls this to change the target number and notify waiting threads.
        """
        with self.condition:
            print(f"🔑 [门卫] 密码已更新为: {new_target}！正在通知所有等待的线程...")
            self.target_number = new_target
            # Wake up all threads that are currently in the wait() state.
            # They will all re-check the condition in their 'while' loop.
            self.condition.notify_all()
