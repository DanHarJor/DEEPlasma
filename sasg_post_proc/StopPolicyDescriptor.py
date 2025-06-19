from pysgpp.extensions.datadriven.learner.LearnerBuilder import LearnerBuilder
StopPolicyDescriptor = LearnerBuilder.StopPolicyDescriptor

class StopPolicyDescriptor(StopPolicyDescriptor):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        