"""Bounded practice extensions, independent of hidden task outcomes."""


class PracticeBudget:
    def __init__(self, initial=5, maximum=20, batch_size=5):
        if any(type(x) is not int for x in (initial, maximum, batch_size)):
            raise ValueError('Attempt budgets must be integers')
        if not 2 <= initial <= maximum <= 20 or not 1 <= batch_size <= 5:
            raise ValueError('Use 2..20 total attempts and extensions of at most five')
        self.initial, self.maximum, self.batch_size = initial, maximum, batch_size
        self.granted = initial
        self.extensions = []

    def status(self, used):
        return {'initial_attempts': self.initial, 'granted_attempts': self.granted,
                'maximum_attempts': self.maximum, 'attempts_started': used,
                'remaining_granted_attempts': max(0, self.granted-used),
                'possible_additional_attempts': self.maximum-self.granted,
                'extension_batch_size': self.batch_size}

    def extend(self, used, active, learned, next_experiments):
        if active or used != self.granted:
            raise ValueError('Finish the current granted block before requesting more attempts')
        if self.granted >= self.maximum:
            raise ValueError('Hard practice-attempt cap reached')
        if not isinstance(learned, str) or not learned.strip():
            raise ValueError('Summarize what was learned, including unresolved uncertainty')
        if not isinstance(next_experiments, list) or not 1 <= len(next_experiments) <= self.batch_size:
            raise ValueError('Supply one to five concrete next experiments')
        if any(not isinstance(x, str) or not x.strip() for x in next_experiments):
            raise ValueError('Each next experiment needs a hypothesis or repeatability test')
        previous = self.granted
        self.granted = min(self.maximum, self.granted+self.batch_size)
        record = {'after_attempt': used, 'previous_grant': previous,
                  'new_grant': self.granted, 'learned': learned,
                  'next_experiments': list(next_experiments),
                  'native_success_used': False,
                  'content_judgment': 'Agent-authored rationale; format and budget checked only'}
        self.extensions.append(record)
        return {**self.status(used), 'extension': record}
