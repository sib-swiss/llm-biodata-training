"""Initialize the Chainlit SQLite database schema using SQLAlchemy."""

import asyncio
from sqlalchemy import (
    Boolean, Column, ForeignKey, Integer, MetaData, String, Table, Text
)
from sqlalchemy.ext.asyncio import create_async_engine

DB_URL = "sqlite+aiosqlite:///./data/chainlit.db"

metadata = MetaData()

Table("users", metadata,
    Column("id", String, primary_key=True),
    Column("identifier", String, nullable=False, unique=True),
    Column("createdAt", String),
    Column("metadata", Text, nullable=False),
)

Table("threads", metadata,
    Column("id", String, primary_key=True),
    Column("createdAt", String),
    Column("name", String),
    Column("userId", String, ForeignKey("users.id", ondelete="CASCADE")),
    Column("userIdentifier", String),
    Column("tags", String),
    Column("metadata", Text),
)

Table("steps", metadata,
    Column("id", String, primary_key=True),
    Column("name", String, nullable=False),
    Column("type", String, nullable=False),
    Column("threadId", String, ForeignKey("threads.id", ondelete="CASCADE"), nullable=False),
    Column("parentId", String),
    Column("streaming", Boolean, nullable=False, default=False),
    Column("waitForAnswer", Boolean),
    Column("isError", Boolean),
    Column("metadata", Text),
    Column("tags", String),
    Column("input", Text),
    Column("output", Text),
    Column("createdAt", String),
    Column("start", String),
    Column("end", String),
    Column("generation", Text),
    Column("showInput", String),
    Column("language", String),
    Column("indent", Integer),
    Column("defaultOpen", Boolean),
    Column("autoCollapse", Boolean),
)

Table("elements", metadata,
    Column("id", String, primary_key=True),
    Column("threadId", String, ForeignKey("threads.id", ondelete="CASCADE")),
    Column("type", String),
    Column("chainlitKey", String),
    Column("url", String),
    Column("objectKey", String),
    Column("name", String, nullable=False),
    Column("display", String),
    Column("size", String),
    Column("language", String),
    Column("page", Integer),
    Column("autoPlay", Boolean),
    Column("playerConfig", Text),
    Column("forId", String),
    Column("mime", String),
    Column("props", Text),
)

Table("feedbacks", metadata,
    Column("id", String, primary_key=True),
    Column("forId", String, nullable=False),
    Column("threadId", String, nullable=False),
    Column("value", Integer, nullable=False),
    Column("comment", Text),
)


async def main():
    engine = create_async_engine(DB_URL)
    async with engine.begin() as conn:
        await conn.run_sync(metadata.create_all)
    await engine.dispose()
    print("Schema created successfully.")

asyncio.run(main())
