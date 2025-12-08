import { useSearchParams } from "react-router";

import {
  Button,
  Dialog,
  DialogContent,
  DialogHeader,
  DialogTitle,
  DialogTrigger,
  Modal,
  ModalOverlay,
} from "@phoenix/components";
import { PlaygroundOptimizerPanel } from "./PlaygroundOptimizerPanel";

export function PlaygroundOptimizerButton() {
  const [searchParams] = useSearchParams();
  const datasetId = searchParams.get("datasetId");

  return (
    <DialogTrigger>
      <Button
        size="S"
        variant="primary"
        isDisabled={!datasetId}
      >
        Optimize
      </Button>
      <ModalOverlay>
        <Modal variant="slideover" size="M">
          <Dialog>
            <DialogContent>
              <DialogHeader>
                <DialogTitle>Prompt Optimizer</DialogTitle>
              </DialogHeader>
              <PlaygroundOptimizerPanel />
            </DialogContent>
          </Dialog>
        </Modal>
      </ModalOverlay>
    </DialogTrigger>
  );
}
