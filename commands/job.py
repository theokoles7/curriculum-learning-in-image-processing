"""Job class."""

from datetime               import datetime
from os                     import makedirs

from matplotlib.pyplot      import legend, plot, savefig, title, xlabel
from numpy                  import argsort, max, median, min, mean, ndarray, std
from numpy.typing           import ArrayLike
from pandas                 import DataFrame, read_csv
from sklearn.metrics        import accuracy_score
from termcolor              import colored
from tqdm                   import tqdm
from torch                  import argmax, no_grad, Tensor
from torch.cuda             import get_device_name, is_available
from torch.nn.functional    import cross_entropy
from torch.optim            import SGD

from curriculums            import curriculums
from datasets               import get_dataset, Cifar10
from models                 import get_model, CNN
from utils                  import ARGS, LOGGER

class Job():
    """Job class."""

    def __init__(self):
        """Initialize job."""
        # Initialize output directory path
        self._output_dir:           str =       f"{ARGS.output_path}/{ARGS.model}/{ARGS.dataset}/{ARGS.curriculum}/{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        # Fetch dataset
        self._dataset:              Cifar10 =   get_dataset(dataset = ARGS.dataset, path = ARGS.dataset_path, batch_size = ARGS.batch_size, curriculum = ARGS.curriculum, by_batch = ARGS.by_batch)
        self._train, self._test =               self._dataset.get_loaders()

        # Fetch model
        self._model:                CNN =       get_model(ARGS.model, self._dataset.channels_in, self._dataset.num_classes, self._dataset.dim)

        # Initialize optimizer
        self._optimizer:            SGD =       SGD(self._model.parameters(), lr = ARGS.learning_rate, weight_decay = 5e-4, momentum = 0.9)

        # Initialize job statistics report
        self._training_report:      DataFrame = DataFrame(columns = ["Epoch", "Train Accuracy", "Train Loss", "Validation Accuracy", "Validation Loss"])
        
        # Ensure output directory exists
        makedirs(self._output_dir, exist_ok = True)
        LOGGER.info(f"Reports will be saved to: {self._output_dir}")

        # Run model from CUDA if available
        if is_available(): self._model = self._model.to("cuda")
        LOGGER.info(f"Using device: {get_device_name()}")

        LOGGER.debug(f"DATASET:\n{self._dataset}\nTRAIN LOADER:\n{vars(self._train)}\nTEST LOADER:\n{vars(self._test)}")
        LOGGER.debug(f"MODEL:\n{self._model}")

    def run(self) -> None:
        """Execute job."""
        LOGGER.info(f"Exeucting job: MODEL {ARGS.model} | DATASET {ARGS.dataset} | CURRICULUM {ARGS.curriculum} | BY BATCH {ARGS.by_batch} | EPOCHS {ARGS.epochs} | BATCH SIZE {ARGS.batch_size}")

        # Conduct training
        train_acc, train_loss, val_acc, val_loss =  self._train_(ARGS.curriculum, by_batch = ARGS.by_batch)

        # Conduct testing
        test_acc, test_loss = self._test_()

        # Save training graph
        self._save_graph(
            path =          f"{self._output_dir}/training_graph.jpg",
            train_acc =     train_acc,
            train_loss =    train_loss,
            val_acc =       val_acc,
            val_loss =      val_loss
        )

        # Record results
        self._record_results(
            path =              f"{ARGS.output_path}/results.csv",
            model =             ARGS.model,
            dataset =           ARGS.dataset,
            curriculum =        ARGS.curriculum if ARGS.curriculum else "none",
            by_batch =          ARGS.by_batch,
            epochs =            ARGS.epochs,
            batch_size =        ARGS.batch_size,
            avg_train_acc =     mean(train_acc),
            avg_train_loss =    mean(train_loss),
            avg_val_acc =       mean(val_acc),
            avg_val_loss =      mean(val_loss),
            test_acc =          test_acc,
            test_loss =         test_loss
        )

        LOGGER.info(f"Job complete: MODEL {ARGS.model} | DATASET {ARGS.dataset} | CURRICULUM {ARGS.curriculum} | EPOCHS {ARGS.epochs} | BATCH SIZE {ARGS.batch_size} | TEST ACCURACY: {test_acc}")

    def _record_results(self,
            path:           str,
            model:          str,
            dataset:        str,
            curriculum:     str,
            by_batch:       str,
            epochs:         str,
            batch_size:     int,
            avg_train_acc:  float,
            avg_train_loss: float,
            avg_val_acc:    float,
            avg_val_loss:   float,
            test_acc:       float,
            test_loss:      float            
        ) -> None:
        """# Record job results in holistic report.

        ## Args:
            * path              (str):      Path at which holistic results report can be located.
            * model             (str):      Model used in job.
            * dataset           (str):      Dataset used in job.
            * curriculum        (str):      Curriculum used in job.
            * by_batch          (str):      Curriculum sorting by batch.
            * epochs            (str):      Epochs job ran for.
            * batch_size        (int):      Batch size used during job.
            * avg_train_acc     (float):    Average train accuracy during job.
            * avg_train_loss    (float):    Average train loss during job.
            * avg_val_acc       (float):    Average validation accuracy during job.
            * avg_val_loss      (float):    Average validation loss during job.
            * test_acc          (float):    Test accuracy at end of job.
            * test_loss         (float):    Test loss at end of job.
        """
        LOGGER.info(f"Recording job results in {path} for {model}, {dataset}, {curriculum}, {by_batch}, {epochs}, {batch_size}")

        # Open file
        file_out = read_csv(path)
        
        # Record job results
        file_out.loc[
            (file_out["MODEL"]      ==  str(model))                 & 
            (file_out["DATASET"]    ==  str(dataset))               & 
            (file_out["CURRICULUM"] ==  str(curriculum))            & 
            (file_out["BY BATCH"]   ==  by_batch)                   &
            (file_out["EPOCHS"]     ==  epochs)                     &
            (file_out["BATCH SIZE"] ==  batch_size),
            ["AVG TRAIN ACCURACY", "AVG TRAIN LOSS", "AVG VALIDATION ACCURACY", "AVG VALIDATION LOSS", "TEST ACCURACY", "TEST LOSS"]] = avg_train_acc, avg_train_loss, avg_val_acc, avg_val_loss, test_acc, test_loss
        
        # Write back
        file_out.to_csv(path, index=False)

    def _save_graph(self,
            path:       str,
            train_acc:  ArrayLike,
            train_loss: ArrayLike,
            val_acc:    ArrayLike,
            val_loss:   ArrayLike
        ) -> None:
        """# Save graph depicting training and validation statistics.

        ## Args:
            * train_acc     (ArrayLike):    Train accuracies.
            * train_loss    (ArrayLike):    Train losses.
            * val_acc       (ArrayLike):    Validation accuracies.
            * val_loss      (ArrayLike):    Validation losses.
        """
        # Log action
        LOGGER.info(f"Saving job graph to {path}")

        # Plot points
        plot(train_acc,     label = "Train Accuracy (%)",       color = "darkviolet")
        plot(train_loss,    label = "Train Loss",               color = "magenta")
        plot(val_acc,       label = "Validation Accuracy (%)",  color = "teal")
        plot(val_loss,      label = "Validatoin Loss",          color = "turquoise")
        
        # Set title, label, & legend
        xlabel('Epoch')
        title(f"MODEL: {ARGS.model} | DATASET: {ARGS.dataset}\nCURRICULUM: {ARGS.curriculum} | BATCH SIZE {ARGS.batch_size}")
        legend()

        # Save figure
        savefig(path)

    def _train_(self, curriculum: str = None, by_batch: bool = False) -> tuple[list, list, list, list]:
        """# Conduct training and validation phases.

        ## Args:
            * curriculum    (str):  Curriculum by which dataset will be sorted. Defaults to None.
            * by_batch      (bool): Sort individual batches, versus entire dataset. Defaults to False.

        ## Returns:
            * tuple[list, list, list, list]:    Four lists containing: training accuracies, training 
                                                loss, validation accuracy, & validation loss, by 
                                                epoch.
        """
        # Log action
        LOGGER.info("Commencing training phase...")
        
        # Initialize running metrics lists
        train_accs, train_losses, val_accs, val_losses = [], [], [], []

        # For every epoch specified
        for epoch in range(1, ARGS.epochs + 1):

            # Initialize progress bar
            with tqdm(
                total =     len(self._train) + len(self._test), 
                desc =      f"Epoch {epoch}/{ARGS.epochs}", 
                leave =     False, 
                colour =    "magenta"
            ) as pbar:
                 
                # TRAIN ---------------------------------------------------------------------------
                # Update progress bar status to "Training"
                pbar.set_postfix(status = colored(text = "Training", color = "cyan"))

                # Put model into training mode
                self._model.train()
                
                # Initialize training metrics
                train_correct = train_total = 0

                # For image:label pairs in train data...
                for images, labels in self._train:
                    
                    # Place on GPU if available
                    if is_available(): images, labels = images.cuda(), labels.cuda()
                    
                    # Reset parameters
                    self._optimizer.zero_grad()
                    
                    # Make predictions & calculate loss
                    train_predictions:  Tensor =    self._model(images)
                    train_loss:         Tensor =    cross_entropy(train_predictions, labels)
                    
                    # Update correct & total
                    train_correct       +=  accuracy_score(labels.cpu(), argmax(train_predictions, dim = 1).cpu().numpy(), normalize = False)
                    train_total         +=  images.size(0)
                    
                    # Back propogation
                    train_loss.backward()
                    self._optimizer.step()
                    
                    # Update progress bar
                    pbar.update(1)
                    
                # Calculate training accuracy & loss
                train_acc:      float =     round((train_correct / train_total) * 100, 4)
                train_loss:     float =     round(train_loss.item(), 4)
                
                # VALIDATE ------------------------------------------------------------------------
                # Update progress bar status to "Validating"
                pbar.set_postfix(status=colored(text = "Validating", color = "yellow"))

                # Put model into evaluation mode
                self._model.eval()

                # Initialize validation metrics
                val_total = val_correct = 0

                # For image:label pairs in test data...
                for images, labels in self._test:

                    # Place on GPU if available
                    if is_available(): images, labels = images.cuda(), labels.cuda()

                    # Make predictions and calculate accuracy
                    with no_grad():
                        
                        # Make predictions & calculate loss
                        val_predictions:    Tensor =    self._model(images)
                        val_loss:           Tensor =    cross_entropy(val_predictions, labels)
                        
                        # Update correct & total
                        val_correct +=  accuracy_score(labels.cpu(), argmax(val_predictions, dim = 1).cpu().numpy(), normalize = False)
                        val_total   +=  images.size(0)

                    # Update progress bar
                    pbar.update(1)
                        
                # Calculate accuracy/loss
                val_acc:    float = round((val_correct / val_total) * 100, 4)
                val_loss:   float = round(val_loss.item(), 4)

                # Record running epoch metrics
                train_accs.append(train_acc)
                train_losses.append(train_loss)
                val_accs.append(val_acc)
                val_losses.append(val_loss)
                self._training_report.loc[epoch-1] = [int(epoch), train_accs[-1], train_losses[-1], val_accs[-1], val_losses[-1]]

                # Update progress bar
                pbar.set_postfix(status=colored("Complete", "green"))
            
            # Log epoch metrics
            LOGGER.info(f"Epoch {epoch:>3}/{ARGS.epochs:>3} | Training Accuracy: {train_accs[-1]:>7.4f}% | Training Loss: {train_losses[-1]:>7.4f} | Validation Accuracy: {val_accs[-1]:>7.4f}% | Validation Loss: {val_losses[-1]:>7.4f}")
        
        # Write distributions to training report & save it
        self._training_report.loc[epoch] =      ["AVERAGE", mean(train_accs),   mean(train_losses),   mean(val_accs),      mean(val_losses)]
        self._training_report.loc[epoch + 1] =  ["MAXIMUM", max(train_accs),    max(train_losses),    max(val_accs),       max(val_losses)]
        self._training_report.loc[epoch + 2] =  ["MEDIAN",  median(train_accs), median(train_losses), median(val_accs),    median(val_losses)]
        self._training_report.loc[epoch + 3] =  ["MINIMUM", min(train_accs),    min(train_losses),    min(val_accs),       min(val_losses)]
        self._training_report.loc[epoch + 4] =  ["STD-DEV", std(train_accs),    std(train_losses),    std(val_accs),       std(val_losses)]
            
        # Return training metrics
        return train_accs, train_losses, val_accs, val_losses

    def _test_(self) -> tuple[float, float]:
        """# Conduct testing phase.

        ## Returns:
            * tuple[float, float]:  Tuple containing testing accuracy and loss.
        """
        LOGGER.info("Commencing testing phase...")

        # Initialize progress bar
        with tqdm(total=len(self._test), desc=colored("Testing", "magenta"), leave = False, colour="cyan") as pbar:

            pbar.set_postfix(status="Testing")

            # Initialize testing metrics
            test_total = test_correct = 0

            # For image:label pairs in test data...
            for images, labels in self._test:

                # Place on GPU if available
                if is_available(): images, labels = images.cuda(), labels.cuda()

                # Feed network
                with no_grad():
                    
                    # Make predictions & calculate loss
                    test_predictions:   Tensor =    self._model(images)
                    test_loss:          Tensor =    cross_entropy(test_predictions, labels)

                    # Update correct & total
                    test_correct    +=  accuracy_score(labels.cpu(), argmax(test_predictions, dim = 1).cpu().numpy(), normalize = False)
                    test_total      +=  images.size(0)

                # Update progress bar
                pbar.update(1)
                
            # Close progress bar
            pbar.close()
                        
            # Calculate accuracy/loss
            test_acc:   float = round((test_correct / test_total) * 100, 4)
            test_loss:  float = round(test_loss.item(), 4)

            # Calculate accuracy
            self._training_report.loc[ARGS.epochs + 5] = ["Test", "Accuracy", test_acc, "Loss", test_loss]
            self._training_report.to_csv(f"{self._output_dir}/training_report.csv")
            LOGGER.info(f"Test Accuracy: {test_acc} | Test Loss: {test_loss}")
            
        # Return accuracy & loss
        return test_acc, test_loss