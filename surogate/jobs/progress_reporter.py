class PrinterJobProgressReporter:
    def __init__(self, job_name: str):
        self.job_name = job_name
        self.current_percentage = 0.0

    def update(self, percentage):
        self.current_percentage = min(percentage, 100.0)
        self.report()

    def report(self):
        print(f"{self.job_name} progress: {self.current_percentage:.2f}%")

    def is_complete(self):
        return self.current_percentage >= 100.0
