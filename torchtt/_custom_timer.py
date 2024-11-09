"""
Custom timer object.
"""
import datetime

class Timer:
    def __init__(self, name=""):
        """
        Timer object. It is meant to be used inside a `with` block.
        The result is printed in `stdout`.

        Examples:

            .. code-block:: python

                import torchtt

                with torchtt.Timer("name"):
                    # do some stuff
                    pass

        Args:
            name (str, optional): the name of the timer. Defaults to "".
        """
        self.name = " \"" + name + "\"" if len(name) > 0 else ""

    def __enter__(self):
        """
        Enter the `with` block. The time point is saved.

        Returns:
            Timer: the timer object.
        """
        self.start = datetime.datetime.now()
        return self

    def __exit__(self, *args):
        """
        Is called when the `with` blocks ends.
        The duration is printed in console.
        """
        self.end = datetime.datetime.now()
        self.interval = (self.end - self.start).total_seconds()
        if self.interval < 1e-6:
            print("Timer%s took %g ns" % (self.name, self.interval*1e9))
        elif self.interval < 1e-3:
            print("Timer%s took %g us" % (self.name, self.interval*1e6))
        elif self.interval < 1:
            print("Timer%s took %g ms" % (self.name, self.interval*1e3))
        else:
            print("Timer%s took %g s" % (self.name, self.interval))
