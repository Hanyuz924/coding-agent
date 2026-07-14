import asyncio
import logging
from collections.abc import Coroutine
from Memory.SelectMemories import read_mem_file, RelevantMemoriesAttachment
logger = logging.getLogger("myagent.memory.prefetch")

class MemoryPrefetch:
    def __init__(self, task:Coroutine):
        self._task :asyncio.Task = asyncio.create_task(task)
        self._consumed = False

    @property
    def settled(self) -> bool:
        return self._task.done()
    
    @property
    def consumed(self) -> bool:
        return self._consumed
    
    def take_task_result(self) -> list:
        self._consumed = True
        if self._task.cancelled():
            return []
        exception = self._task.exception()
        if exception:
            logger.exception("[memory_prefetch] task failed", exc_info=exception)
            return []
        result = self._task.result()
        if result:
            mem_files = read_mem_file(result)
            if mem_files:
                attachment = RelevantMemoriesAttachment(memories=mem_files)
                rendered = attachment.render()
                logger.debug(f"[memory_prefetch] returning {len(mem_files)} memory file(s)")
                return rendered
            logger.debug("[memory_prefetch] no memory files loaded from selected filenames")
        else:
            logger.debug("[memory_prefetch] no filenames selected")
        return []
