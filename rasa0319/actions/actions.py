from typing import Any, Dict, List, Text

from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher


class ActionCheckSufficientFunds(Action):
    def name(self) -> Text:
        return "action_check_sufficient_funds"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:



        balance = tracker.get_slot("balance")
        transfer_amount = tracker.get_slot("amount")

        has_sufficient_funds = transfer_amount <= balance

        return [SlotSet("has_sufficient_funds", has_sufficient_funds)]


class ActionTransfer(Action):
    def name(self) -> Text:
        return "action_transfer"

    def run(
        self,
        dispatcher: CollectingDispatcher,
        tracker: Tracker,
        domain: Dict[Text, Any],
    ) -> List[Dict[Text, Any]]:

        # 执行转账，更新余额balance
        # 不需要再判断 余额 和 转账金额的大小，能执行到这里，肯定是已经判断过有足够余额

        balance = tracker.get_slot("balance")
        transfer_amount = tracker.get_slot("amount")


        return [SlotSet("balance", balance-transfer_amount)]