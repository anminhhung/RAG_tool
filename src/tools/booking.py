from typing import Optional

from llama_index.core.tools import FunctionTool
from llama_index.core.bridge.pydantic import BaseModel


class Booking(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    phone: Optional[str] = None
    date: Optional[str] = None
    time: Optional[str] = None


def load_booking_tools() -> list[FunctionTool]:

    bookings = {}

    def get_booking_state(user_id: str) -> str:
        """Get the current state of a booking for a given booking ID."""
        try:
            return str(bookings[user_id].dict())
        except KeyError:
            return f"Booking ID {user_id} not found"

    def update_booking(user_id: str, property: str, value: str) -> str:
        """Update a property of a booking for a given booking ID. Only enter details that are explicitly provided."""
        booking = bookings[user_id]
        setattr(booking, property, value)
        return f"Booking ID {user_id} updated with {property} = {value}"

    def create_booking(user_id: str) -> str:
        """Create a new booking and return the booking ID."""
        bookings[user_id] = Booking()
        return "Booking created, but not yet confirmed. Please provide your name, email, phone, date, and time."

    def confirm_booking(user_id: str) -> str:
        """Confirm a booking for a given booking ID."""
        booking = bookings[user_id]

        if booking.name is None:
            raise ValueError("Please provide your name.")

        if booking.email is None:
            raise ValueError("Please provide your email.")

        if booking.phone is None:
            raise ValueError("Please provide your phone number.")

        if booking.date is None:
            raise ValueError("Please provide the date of your booking.")

        if booking.time is None:
            raise ValueError("Please provide the time of your booking.")

        return f"Booking ID {user_id} confirmed!"

    # create tools for each function
    get_booking_state_tool = FunctionTool.from_defaults(
        fn=get_booking_state,
        description="Get the current state of a booking for a given booking ID.",
    )
    update_booking_tool = FunctionTool.from_defaults(
        fn=update_booking,
        description="Update a property of a booking for a given booking ID.",
    )
    create_booking_tool = FunctionTool.from_defaults(
        fn=create_booking,
        return_direct=True,
        description="Create a new booking and return the booking ID.",
    )
    confirm_booking_tool = FunctionTool.from_defaults(
        fn=confirm_booking,
        return_direct=True,
        description="Confirm a booking for a given booking ID.",
    )

    return [
        get_booking_state_tool,
        update_booking_tool,
        create_booking_tool,
        confirm_booking_tool,
    ]
